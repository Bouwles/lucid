using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace Lucid.Recorder
{
    /// <summary>One logic step as it appears in steps.jsonl.</summary>
    public struct StepRecord
    {
        public int Step;
        public float Time;
        public UnifiedAction Action;
        public GameStateSnapshot State;
        public bool EpisodeEnd;
    }

    /// <summary>Session metadata: session.json, and the shared fields of every meta.json.</summary>
    public sealed class SessionInfo
    {
        public readonly string SessionId;
        public readonly string GameId;
        public readonly string Folder;
        public readonly DateTime CreatedUtc;
        public readonly string GameBuildVersion;
        public readonly string UnityVersion;
        public readonly int Width;
        public readonly int Height;
        public readonly float LogicTimestep;
        public readonly float? LookScale;

        public SessionInfo(string sessionId, string gameId, string folder, DateTime createdUtc, string gameBuildVersion,
            string unityVersion, int width, int height, float logicTimestep, float? lookScale)
        {
            SessionId = sessionId;
            GameId = gameId;
            Folder = folder;
            CreatedUtc = createdUtc;
            GameBuildVersion = gameBuildVersion;
            UnityVersion = unityVersion;
            Width = width;
            Height = height;
            LogicTimestep = logicTimestep;
            LookScale = lookScale;
        }

        public void WriteSessionFile()
        {
            var w = new JsonWriter();
            w.BeginObject()
                .Key("format_version").Value(RecorderVersion.FormatVersion)
                .Key("session_id").Value(SessionId)
                .Key("game_id").Value(GameId)
                .Key("created_utc").Value(CreatedUtc.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", CultureInfo.InvariantCulture))
                .Key("recorder_version").Value(RecorderVersion.Value)
                .Key("game_build_version").Value(GameBuildVersion)
                .Key("unity_version").Value(UnityVersion)
                .Key("recording_resolution").BeginArray().Value(Width).Value(Height).EndArray()
                .Key("logic_timestep").Value(LogicTimestep)
                .Key("frame_format").Value("png")
                .EndObject();
            EpisodeWriter.WriteTextAtomic(Path.Combine(Folder, "session.json"), w.ToString());
        }
    }

    /// <summary>
    /// One episode's bookkeeping. The recorder fills the public fields on the game thread before it
    /// queues the episode end; the writer thread owns the internal fields.
    /// </summary>
    public sealed class EpisodeInfo
    {
        public readonly int Index;
        public readonly string EpisodeId;
        public readonly string Folder;
        public readonly int Seed;
        public readonly string ActionSourceName;
        public int NumSteps;
        public Dictionary<string, object> ActionSourceSettings;

        internal bool Opened;
        internal string FramesFolder;
        internal StreamWriter Steps;
        internal int NextStep;
        internal int LinesWritten;
        internal int FramesWritten;
        internal int LinesSinceFlush;

        readonly object _lock = new object();
        string _corruptionReason;

        public EpisodeInfo(int index, string episodeId, string folder, int seed, string actionSourceName)
        {
            Index = index;
            EpisodeId = episodeId;
            Folder = folder;
            Seed = seed;
            ActionSourceName = actionSourceName;
        }

        /// <summary>Keeps the first reason; safe to call from any thread.</summary>
        public void MarkCorrupted(string reason)
        {
            lock (_lock)
            {
                if (_corruptionReason == null) _corruptionReason = reason;
            }
        }

        public string CorruptionReason
        {
            get
            {
                lock (_lock) return _corruptionReason;
            }
        }
    }

    /// <summary>
    /// Writes frames, step lines and meta.json on a background thread. The queue is bounded:
    /// when it is full, <see cref="EnqueueStep"/> blocks the game thread instead of dropping a frame.
    /// The recorder pauses game logic before it gets that far.
    /// </summary>
    public sealed class EpisodeWriter
    {
        enum JobKind : byte
        {
            Step,
            EndEpisode,
        }

        struct Job
        {
            public JobKind Kind;
            public EpisodeInfo Episode;
            public StepRecord Step;
            public byte[] Pixels;
        }

        const int FlushEveryLines = 60;
        static readonly UTF8Encoding Utf8NoBom = new UTF8Encoding(false);

        readonly SessionInfo _session;
        readonly BufferPool _pool;
        readonly int _width;
        readonly int _height;
        readonly bool _flipRows;
        readonly byte[] _rgb;
        readonly JsonWriter _json = new JsonWriter(512);
        readonly Job[] _ring;
        readonly object _lock = new object();
        readonly Thread _thread;
        int _head;
        int _count;
        bool _completing;
        bool _faulted;
        int _writeErrors;
        int _framesWritten;
        int _episodesWritten;
        volatile string _lastError;

        internal EpisodeWriter(SessionInfo session, BufferPool pool, int capacity, bool flipRows)
        {
            _session = session;
            _pool = pool;
            _width = session.Width;
            _height = session.Height;
            _flipRows = flipRows;
            _rgb = new byte[_width * _height * 3];
            _ring = new Job[Mathf.Max(4, capacity)];
            _thread = new Thread(Run) { Name = "LUCID episode writer", IsBackground = true };
        }

        public int Capacity => _ring.Length;
        public int WriteErrors => Volatile.Read(ref _writeErrors);
        public int FramesWritten => Volatile.Read(ref _framesWritten);
        public int EpisodesWritten => Volatile.Read(ref _episodesWritten);
        public string LastError => _lastError;

        public int QueueDepth
        {
            get
            {
                lock (_lock) return _count;
            }
        }

        public void Start() => _thread.Start();

        /// <param name="pixels">RGBA32 readback, bottom row first, from the pool; null if the readback failed.</param>
        internal void EnqueueStep(EpisodeInfo episode, in StepRecord step, byte[] pixels)
        {
            Enqueue(new Job { Kind = JobKind.Step, Episode = episode, Step = step, Pixels = pixels });
        }

        /// <summary>Closes steps.jsonl and writes meta.json. Queue after the episode's last step.</summary>
        internal void EnqueueEpisodeEnd(EpisodeInfo episode)
        {
            Enqueue(new Job { Kind = JobKind.EndEpisode, Episode = episode });
        }

        /// <summary>Writes everything still queued, then stops the thread. Blocks until done.</summary>
        public void CompleteAndWait()
        {
            lock (_lock)
            {
                _completing = true;
                Monitor.PulseAll(_lock);
            }
            if (_thread.IsAlive) _thread.Join();
        }

        void Enqueue(in Job job)
        {
            lock (_lock)
            {
                while (_count == _ring.Length && !_faulted) Monitor.Wait(_lock);
                if (_faulted || _completing)
                {
                    job.Episode?.MarkCorrupted("The writer thread had stopped; data was lost.");
                    _pool.Return(job.Pixels);
                    return;
                }
                _ring[(_head + _count) % _ring.Length] = job;
                _count++;
                Monitor.PulseAll(_lock);
            }
        }

        void Run()
        {
            try
            {
                while (true)
                {
                    Job job;
                    lock (_lock)
                    {
                        while (_count == 0 && !_completing) Monitor.Wait(_lock);
                        if (_count == 0) return;
                        job = _ring[_head];
                        _ring[_head] = default;
                        _head = (_head + 1) % _ring.Length;
                        _count--;
                        Monitor.PulseAll(_lock);
                    }

                    if (job.Kind == JobKind.Step) WriteStep(job);
                    else FinishEpisode(job.Episode);
                }
            }
            catch (Exception e)
            {
                Fail(null, e);
                lock (_lock)
                {
                    _faulted = true;
                    while (_count > 0)
                    {
                        Job left = _ring[_head];
                        left.Episode?.MarkCorrupted("The writer thread stopped: " + e.Message);
                        _pool.Return(left.Pixels);
                        _ring[_head] = default;
                        _head = (_head + 1) % _ring.Length;
                        _count--;
                    }
                    Monitor.PulseAll(_lock);
                }
            }
        }

        void WriteStep(in Job job)
        {
            EpisodeInfo ep = job.Episode;
            try
            {
                Open(ep);
                int step = job.Step.Step;
                if (step != ep.NextStep) ep.MarkCorrupted($"Step {step} arrived when step {ep.NextStep} was expected.");
                ep.NextStep = step + 1;

                if (job.Pixels != null)
                {
                    ToRgbTopFirst(job.Pixels);
                    byte[] png = ImageConversion.EncodeArrayToPNG(_rgb, GraphicsFormat.R8G8B8_SRGB,
                        (uint)_width, (uint)_height, (uint)(_width * 3));
                    if (png == null) throw new IOException($"PNG encoding failed for step {step}.");
                    File.WriteAllBytes(Path.Combine(ep.FramesFolder, step.ToString("D6", CultureInfo.InvariantCulture) + ".png"), png);
                    ep.FramesWritten++;
                    Interlocked.Increment(ref _framesWritten);
                }

                WriteStepLine(ep, job.Step);
                ep.LinesWritten++;
            }
            catch (Exception e)
            {
                Fail(ep, e);
            }
            finally
            {
                _pool.Return(job.Pixels);
            }
        }

        void Open(EpisodeInfo ep)
        {
            if (ep.Opened) return;
            ep.Opened = true;
            ep.FramesFolder = Path.Combine(ep.Folder, "frames");
            Directory.CreateDirectory(ep.FramesFolder);
            // CreateNew: never overwrite an existing recording.
            var stream = new FileStream(Path.Combine(ep.Folder, "steps.jsonl"), FileMode.CreateNew, FileAccess.Write,
                FileShare.Read, 1 << 16);
            ep.Steps = new StreamWriter(stream, Utf8NoBom) { NewLine = "\n" };
        }

        // Drops alpha and fixes row order. GPU readbacks arrive bottom row first; whether the
        // encoder needs them reversed was measured at startup by DetectRowFlip.
        void ToRgbTopFirst(byte[] rgba)
        {
            int srcStride = _width * 4;
            int dstStride = _width * 3;
            for (int y = 0; y < _height; y++)
            {
                int s = (_flipRows ? _height - 1 - y : y) * srcStride;
                int d = y * dstStride;
                for (int x = 0; x < _width; x++)
                {
                    _rgb[d] = rgba[s];
                    _rgb[d + 1] = rgba[s + 1];
                    _rgb[d + 2] = rgba[s + 2];
                    d += 3;
                    s += 4;
                }
            }
        }

        void WriteStepLine(EpisodeInfo ep, in StepRecord s)
        {
            _json.Clear();
            _json.BeginObject()
                .Key("step").Value(s.Step)
                .Key("time").Value(s.Time)
                .Key("buttons").Value((int)s.Action.Buttons)
                .Key("look").BeginArray().Value(s.Action.LookX).Value(s.Action.LookY).EndArray()
                .Key("state").BeginObject()
                    .Key("lane").Value(s.State.Lane)
                    .Key("x").Value(s.State.X)
                    .Key("y").Value(s.State.Y)
                    .Key("vertical").Value(GameStateSnapshot.ToWire(s.State.Vertical))
                    .Key("alive").Value(s.State.Alive)
                    .Key("speed").Value(s.State.Speed)
                    .Key("distance").Value(s.State.Distance)
                .EndObject()
                .Key("episode_end").Value(s.EpisodeEnd)
                .EndObject();
            ep.Steps.Write(_json.ToString());
            ep.Steps.Write('\n');
            if (++ep.LinesSinceFlush >= FlushEveryLines || s.EpisodeEnd)
            {
                ep.Steps.Flush();
                ep.LinesSinceFlush = 0;
            }
        }

        void FinishEpisode(EpisodeInfo ep)
        {
            if (!ep.Opened && ep.NumSteps == 0) return;

            if (ep.Steps != null)
            {
                try
                {
                    ep.Steps.Flush();
                    ep.Steps.Dispose();
                }
                catch (Exception e)
                {
                    Fail(ep, e);
                }
                ep.Steps = null;
            }

            if (ep.LinesWritten != ep.NumSteps || ep.FramesWritten != ep.NumSteps)
                ep.MarkCorrupted($"{ep.NumSteps} steps recorded but {ep.LinesWritten} lines and {ep.FramesWritten} frames written.");

            try
            {
                Directory.CreateDirectory(ep.Folder);
                WriteMeta(ep);
                Interlocked.Increment(ref _episodesWritten);
            }
            catch (Exception e)
            {
                Fail(ep, e);
            }
        }

        void WriteMeta(EpisodeInfo ep)
        {
            string reason = ep.CorruptionReason;
            _json.Clear();
            _json.BeginObject()
                .Key("format_version").Value(RecorderVersion.FormatVersion)
                .Key("episode_id").Value(ep.EpisodeId)
                .Key("session_id").Value(_session.SessionId)
                .Key("game_id").Value(_session.GameId)
                .Key("seed").Value(ep.Seed)
                .Key("game_build_version").Value(_session.GameBuildVersion)
                .Key("recorder_version").Value(RecorderVersion.Value)
                .Key("action_source").BeginObject()
                    .Key("name").Value(ep.ActionSourceName)
                    .Key("settings").Any(ep.ActionSourceSettings ?? new Dictionary<string, object>())
                .EndObject()
                .Key("recording_resolution").BeginArray().Value(_session.Width).Value(_session.Height).EndArray()
                .Key("logic_timestep").Value(_session.LogicTimestep)
                .Key("look_scale").Value(_session.LookScale)
                .Key("num_steps").Value(ep.NumSteps)
                .Key("corrupted").Value(reason != null)
                .Key("corruption_reason").Value(reason)
                .EndObject();
            WriteTextAtomic(Path.Combine(ep.Folder, "meta.json"), _json.ToString());
        }

        void Fail(EpisodeInfo ep, Exception e)
        {
            string reason = e.GetType().Name + ": " + e.Message;
            Interlocked.Increment(ref _writeErrors);
            _lastError = reason;
            bool firstForEpisode = ep != null && ep.CorruptionReason == null;
            ep?.MarkCorrupted(reason);
            // Debug.Log is thread-safe. Log once per episode so a full disk does not flood the console.
            if (ep == null || firstForEpisode)
                Debug.LogError("[LUCID] Write failed" + (ep != null ? " in " + ep.EpisodeId : "") + ": " + reason);
        }

        internal static void WriteTextAtomic(string path, string text)
        {
            string tmp = path + ".tmp";
            File.WriteAllText(tmp, text + "\n", Utf8NoBom);
            if (File.Exists(path)) File.Delete(path);
            File.Move(tmp, path);
        }

        /// <summary>
        /// Whether readback rows must be reversed before encoding so that PNG row 0 is the top.
        /// GPU readbacks of a RenderTexture arrive bottom row first on every graphics API (Unity's
        /// texture convention). Whether ImageConversion reverses rows itself is not documented, so
        /// encode a 1x2 image and read the first PNG row back.
        /// </summary>
        public static bool DetectRowFlip()
        {
            try
            {
                byte[] pixels = { 255, 0, 0, 0, 0, 255 }; // array row 0 red, row 1 blue
                byte[] png = ImageConversion.EncodeArrayToPNG(pixels, GraphicsFormat.R8G8B8_SRGB, 1, 2, 3);
                // Row 0 of a 1-pixel-wide PNG reads back raw whatever the filter type, since every
                // PNG filter predicts from the left and upper neighbours, which are zero there.
                byte[] firstRow = InflatePngData(png, 4);
                bool encoderKeepsArrayOrder = firstRow[1] > 127;
                return encoderKeepsArrayOrder;
            }
            catch (Exception e)
            {
                Debug.LogWarning("[LUCID] Could not detect PNG row order (" + e.Message +
                                 "); assuming ImageConversion follows Unity's bottom-up convention. Check that frames are upright.");
                return false;
            }
        }

        static byte[] InflatePngData(byte[] png, int count)
        {
            var idat = new MemoryStream();
            int p = 8; // PNG signature
            while (p + 8 <= png.Length)
            {
                int length = (png[p] << 24) | (png[p + 1] << 16) | (png[p + 2] << 8) | png[p + 3];
                if (png[p + 4] == 'I' && png[p + 5] == 'D' && png[p + 6] == 'A' && png[p + 7] == 'T')
                    idat.Write(png, p + 8, length);
                p += 12 + length;
            }
            idat.Position = 2; // skip the zlib header; DeflateStream reads raw deflate data
            using (var inflate = new DeflateStream(idat, CompressionMode.Decompress))
            {
                var result = new byte[count];
                int read = 0;
                while (read < count)
                {
                    int n = inflate.Read(result, read, count - read);
                    if (n <= 0) throw new InvalidDataException("PNG image data too short.");
                    read += n;
                }
                return result;
            }
        }
    }
}
