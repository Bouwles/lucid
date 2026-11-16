using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Lucid.Recorder
{
    /// <summary>
    /// Records one frame, action and state line per logic step and runs episodes back to back.
    /// Per FixedUpdate: render and read back frame N, read the state, ask the action source for
    /// action N, apply it to the game, queue the step. Runs before every game script, so frame N
    /// shows the world before action N and its effect first appears in frame N+1.
    /// </summary>
    [DefaultExecutionOrder(-10000)]
    [DisallowMultipleComponent]
    [AddComponentMenu("LUCID/Lucid Recorder")]
    public sealed class LucidRecorder : MonoBehaviour
    {
        [SerializeField] RecorderConfig config;

        [Tooltip("Component that implements IGameAdapter and IEpisodeControl.")]
        [SerializeField] MonoBehaviour gameAdapter;

        [Tooltip("Component that implements IActionSource: HumanActionSource, an AutoPilotBase subclass or ReplayActionSource.")]
        [SerializeField] MonoBehaviour actionSource;

        [Tooltip("Camera whose view is recorded. Defaults to Camera.main.")]
        [SerializeField] Camera sourceCamera;

        [Tooltip("Optional camera used only for recording. If empty, a copy of the source camera is created as its child.")]
        [SerializeField] Camera recordingCamera;

        [Tooltip("Start a session as soon as play mode or the build starts.")]
        [SerializeField] bool startOnPlay;

        enum Phase
        {
            Idle,
            Starting,   // BeginEpisode called, waiting for the game to be ready; logic paused
            Recording,
            Ending,     // final step recorded; next episode starts in Update
        }

        [Flags]
        enum PauseReason
        {
            None = 0,
            Episode = 1,
            Backpressure = 2,
        }

        struct PendingFrame
        {
            public AsyncGPUReadbackRequest Request;
            public EpisodeInfo Episode;
            public StepRecord Step;
        }

#if UNITY_EDITOR
        /// <summary>Raised in Start, before startOnPlay is checked. Editor tools use it to set up a run.</summary>
        public static event Action<LucidRecorder> EditorStarting;
#endif

        public event Action<string> SessionFinished;

        IGameAdapter _adapter;
        IEpisodeControl _control;
        IActionSource _source;
        IEpisodeAwareSource _sourceHooks;
        ReplayActionSource _replay;

        Camera _cam;
        bool _ownsCamera;
        RenderTexture _savedTarget;
        int _savedCullingMask;
        bool _savedEnabled;
        RenderTexture _rt;

        SessionInfo _session;
        EpisodeWriter _writer;
        BufferPool _pool;
        PendingFrame[] _pending;
        int _pendingHead;
        int _pendingCount;

        System.Random _seedRng;
        Phase _phase;
        PauseReason _pause;
        float _savedTimeScale = 1f;
        float _dt;

        EpisodeInfo _episode;
        int _step;
        int _deadSteps;
        int _holdSteps;
        int _maxSteps;
        int _episodeBeginFrame;
        int _episodesStarted;
        int _episodesCompleted;
        int _readbackErrors;
        long _framesCaptured;
        float _sessionStartRealtime;
        bool _stopAfterEpisode;

        int _lastFramesWritten;
        int _lastWriteErrors;
        string _lastError;

        public RecorderConfig Config
        {
            get => config;
            set => config = value;
        }

        public MonoBehaviour GameAdapterComponent
        {
            get => gameAdapter;
            set => gameAdapter = value;
        }

        public MonoBehaviour ActionSourceComponent
        {
            get => actionSource;
            set => actionSource = value;
        }

        /// <summary>Replaces the config's output folder for the next session (used by the determinism test).</summary>
        public string OutputFolderOverride { get; set; }

        /// <summary>Replaces config.maxEpisodes when above 0.</summary>
        public int MaxEpisodesOverride { get; set; }

        public bool IsRecording => _phase != Phase.Idle;
        public bool IsLogicPaused => _pause != PauseReason.None;
        public int EpisodesCompleted => _episodesCompleted;
        public int CurrentEpisodeStep => _step;
        public long FramesCaptured => _framesCaptured;
        public int FramesWritten => _writer != null ? _writer.FramesWritten : _lastFramesWritten;
        public int QueueDepth => _writer != null ? _writer.QueueDepth : 0;
        public int QueueCapacity => _writer != null ? _writer.Capacity : 0;
        public int PendingReadbacks => _pendingCount;
        public int WriteErrors => _readbackErrors + (_writer != null ? _writer.WriteErrors : _lastWriteErrors);
        public string LastError => (_writer != null ? _writer.LastError : null) ?? _lastError;
        public string SessionDirectory => _session?.Folder;

        void Awake()
        {
#if !UNITY_EDITOR && !LUCID_RECORDING
            Debug.LogWarning("[LUCID] Recorder disabled: this build was made without the LUCID_RECORDING define.", this);
            enabled = false;
#endif
        }

        void Start()
        {
#if UNITY_EDITOR
            EditorStarting?.Invoke(this);
#endif
            if (startOnPlay && !IsRecording) StartRecording();
        }

        void OnDisable() => StopRecording();

        void OnApplicationQuit() => StopRecording();

        /// <summary>Starts a new session. Returns false (and logs why) if the setup is incomplete.</summary>
        public bool StartRecording()
        {
            if (_phase != Phase.Idle) return true;
            if (!isActiveAndEnabled)
            {
                Fail("The recorder component is disabled.");
                return false;
            }

            string error = Validate();
            if (error != null)
            {
                Fail(error);
                return false;
            }

            try
            {
                BeginSession();
            }
            catch (Exception e)
            {
                Fail("Could not start recording: " + e.Message);
                if (_writer != null)
                {
                    _writer.CompleteAndWait();
                    _writer = null;
                }
                TearDown();
                return false;
            }

            BeginNextEpisode();
            return true;
        }

        /// <summary>Ends the session once the current episode has finished.</summary>
        public void StopAfterCurrentEpisode() => _stopAfterEpisode = true;

        /// <summary>
        /// Stops now: waits for outstanding readbacks, marks the last recorded step as the episode
        /// end, writes everything still queued and closes all files.
        /// </summary>
        public void StopRecording()
        {
            if (_phase == Phase.Idle) return;
            _phase = Phase.Idle;

            try
            {
                if (_pendingCount > 0)
                {
                    AsyncGPUReadback.WaitAllRequests();
                    DrainReadbacks(true);
                }
            }
            catch (Exception e)
            {
                Debug.LogException(e, this);
            }

            if (_writer != null)
            {
                _writer.CompleteAndWait();
                _lastFramesWritten = _writer.FramesWritten;
                _lastWriteErrors = _writer.WriteErrors;
                _lastError = _writer.LastError ?? _lastError;
                _writer = null;
            }

            TearDown();
            SetPause(PauseReason.Episode | PauseReason.Backpressure, false);

            string folder = _session?.Folder;
            Debug.Log($"[LUCID] Stopped. Episodes: {_episodesCompleted}, frames written: {_lastFramesWritten}, " +
                      $"write errors: {WriteErrors}. {folder}", this);
            SessionFinished?.Invoke(folder);
        }

        string Validate()
        {
            if (config == null) return "No RecorderConfig assigned.";

            _adapter = gameAdapter as IGameAdapter;
            if (_adapter == null) return "The game adapter must be a component that implements IGameAdapter.";
            _control = gameAdapter as IEpisodeControl;
            if (_control == null)
                return "The game adapter must also implement IEpisodeControl, so the recorder can restart episodes with a seed.";

            _source = actionSource as IActionSource;
            if (_source == null) return "The action source must be a component that implements IActionSource.";
            _sourceHooks = _source as IEpisodeAwareSource;
            _replay = _source as ReplayActionSource;
            if (_replay != null)
            {
                try
                {
                    _replay.EnsureLoaded();
                }
                catch (Exception e)
                {
                    return "The ReplayActionSource could not load its episode: " + e.Message;
                }
                if (_replay.StepCount == 0) return "The replayed episode has no steps.";
            }

            if (!SystemInfo.supportsAsyncGPUReadback)
                return "This platform or graphics API does not support AsyncGPUReadback.";
            if (Time.fixedDeltaTime <= 0f) return "Time.fixedDeltaTime must be positive.";

            string gameId = _adapter.GameId;
            if (string.IsNullOrEmpty(gameId) || gameId.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || gameId.Contains(" "))
                return $"Invalid game id '{gameId}'. Use the id from configs/games.yaml.";
            return null;
        }

        void BeginSession()
        {
            _dt = Time.fixedDeltaTime;
            int width = config.width;
            int height = config.height;

            string root = string.IsNullOrEmpty(OutputFolderOverride)
                ? config.ResolveOutputFolder()
                : Path.GetFullPath(OutputFolderOverride);
            DateTime now = DateTime.UtcNow;
            string sessionId = _adapter.GameId + "_" + now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            string folder = Path.Combine(root, sessionId);
            if (Directory.Exists(folder))
                throw new IOException($"{folder} already exists. Wait a second and start again.");

            string build = string.IsNullOrEmpty(config.gameBuildVersion) ? Application.version : config.gameBuildVersion;
            float? lookScale = config.actionMapper != null ? config.actionMapper.LookScaleForMeta : null;
            _session = new SessionInfo(sessionId, _adapter.GameId, folder, now, build, Application.unityVersion,
                width, height, _dt, lookScale);
            Directory.CreateDirectory(Path.Combine(folder, "episodes"));
            _session.WriteSessionFile();

            SetupCamera(width, height);

            int capacity = Mathf.Max(4, config.queueCapacity);
            int ringSize = Mathf.Max(2, config.maxPendingReadbacks + 1);
            _pool = new BufferPool(width * height * 4, capacity + ringSize + 1);
            bool flip = config.rowOrder == RecorderConfig.RowOrder.Flip ||
                        (config.rowOrder == RecorderConfig.RowOrder.Auto && EpisodeWriter.DetectRowFlip());
            _writer = new EpisodeWriter(_session, _pool, capacity, flip);
            _writer.Start();
            _pending = new PendingFrame[ringSize];
            _pendingHead = 0;
            _pendingCount = 0;

            _seedRng = new System.Random(config.sessionSeed != 0 ? config.sessionSeed : (int)(now.Ticks & 0x7FFFFFFF));
            if (_source is AutoPilotBase autopilot) autopilot.Configure(config.autopilot);
            if (_source is HumanActionSource human && config.actionMapper != null) human.Mapper = config.actionMapper;

            _episode = null;
            _episodesStarted = 0;
            _episodesCompleted = 0;
            _framesCaptured = 0;
            _readbackErrors = 0;
            _lastFramesWritten = 0;
            _lastWriteErrors = 0;
            _lastError = null;
            _stopAfterEpisode = false;
            _sessionStartRealtime = Time.realtimeSinceStartup;

            Debug.Log($"[LUCID] Recording {sessionId} to {folder} (PNG row flip: {flip})", this);
        }

        void SetupCamera(int width, int height)
        {
            if (recordingCamera != null)
            {
                _cam = recordingCamera;
                _ownsCamera = false;
                _savedTarget = _cam.targetTexture;
                _savedCullingMask = _cam.cullingMask;
                _savedEnabled = _cam.enabled;
            }
            else
            {
                Camera src = sourceCamera != null ? sourceCamera : Camera.main;
                if (src == null)
                    throw new InvalidOperationException("No camera to record: assign a source camera or tag one MainCamera.");
                var go = new GameObject("LUCID Recording Camera");
                _cam = go.AddComponent<Camera>();
                _cam.CopyFrom(src);
                go.transform.SetParent(src.transform, true);
                _ownsCamera = true;
            }

            // sRGB target: the readback holds the same gamma-encoded values the player sees.
            var desc = new RenderTextureDescriptor(width, height, GraphicsFormat.R8G8B8A8_SRGB, 24) { msaaSamples = 1 };
            _rt = new RenderTexture(desc) { name = "LUCID Recording" };
            _rt.Create();
            _cam.targetTexture = _rt;
            _cam.ResetAspect();
            _cam.cullingMask &= ~config.excludedLayers.value;
            _cam.enabled = false; // rendered by hand, once per logic step
        }

        void TearDown()
        {
            if (_cam != null)
            {
                if (_ownsCamera)
                {
                    _cam.targetTexture = null;
                    Destroy(_cam.gameObject);
                }
                else
                {
                    _cam.targetTexture = _savedTarget;
                    _cam.cullingMask = _savedCullingMask;
                    _cam.enabled = _savedEnabled;
                }
                _cam = null;
            }
            if (_rt != null)
            {
                _rt.Release();
                Destroy(_rt);
                _rt = null;
            }
            _pending = null;
            _pendingHead = 0;
            _pendingCount = 0;
            _pool = null;
        }

        void BeginNextEpisode()
        {
            int seed = _replay != null && _replay.HasSeed ? _replay.Seed : _seedRng.Next();
            int index = _episodesStarted++;
            string id = "ep_" + index.ToString("D6", CultureInfo.InvariantCulture);
            _episode = new EpisodeInfo(index, id, Path.Combine(_session.Folder, "episodes", id), seed, _source.Name);
            _step = 0;
            _deadSteps = 0;
            _holdSteps = Mathf.Max(1, Mathf.RoundToInt(config.gameOverHoldSeconds / _dt));
            _maxSteps = config.maxEpisodeSeconds > 0f
                ? Mathf.Max(1, Mathf.RoundToInt(config.maxEpisodeSeconds / _dt))
                : int.MaxValue;
            if (_replay != null) _maxSteps = _replay.StepCount;

            // Logic stays paused from here until the first step, so the number of unrecorded ticks
            // between the reset and frame 0 is always zero, whatever the frame rate.
            SetPause(PauseReason.Episode, true);
            _phase = Phase.Starting;
            _episodeBeginFrame = Time.frameCount;
            try
            {
                _control.BeginEpisode(seed);
                _sourceHooks?.BeginEpisode(seed);
            }
            catch (Exception e)
            {
                Debug.LogException(e, this);
                Fail("BeginEpisode threw an exception; recording stopped.");
                StopRecording();
            }
        }

        void Update()
        {
            if (_phase == Phase.Idle) return;
            DrainReadbacks(false);
            UpdateBackpressure();

            if (_phase == Phase.Starting)
            {
                // Waiting one rendered frame lets Destroy() calls from the reset take effect.
                if (Time.frameCount > _episodeBeginFrame && _control.EpisodeReady)
                {
                    _phase = Phase.Recording;
                    SetPause(PauseReason.Episode, false);
                }
            }
            else if (_phase == Phase.Ending)
            {
                if (SessionShouldEnd()) StopRecording();
                else BeginNextEpisode();
            }
        }

        void FixedUpdate()
        {
            if (_phase == Phase.Idle) return;
            if (_phase != Phase.Recording)
            {
                // Ticks that still run in the render frame where an episode ended. They are not
                // recorded and the world is reset next, so just keep the input neutral.
                _adapter.Apply(default);
                return;
            }

            DrainReadbacks(false);
            if (_pendingCount == _pending.Length)
            {
                _pending[_pendingHead].Request.WaitForCompletion();
                DrainReadbacks(false);
            }

            _cam.Render();
            // Native format (R8G8B8A8_SRGB): asking for a different format could make Unity convert
            // sRGB to linear during the copy and darken the frames.
            AsyncGPUReadbackRequest request = AsyncGPUReadback.Request(_rt, 0);

            GameStateSnapshot state = _adapter.ReadState();
            UnifiedAction action = _source.GetAction(state).Sanitized();
            if (!state.Alive)
            {
                action = default;
                _deadSteps++;
            }
            bool terminal = _deadSteps >= _holdSteps || _step >= _maxSteps - 1;
            _adapter.Apply(action);

            int slot = (_pendingHead + _pendingCount) % _pending.Length;
            _pending[slot].Request = request;
            _pending[slot].Episode = _episode;
            _pending[slot].Step = new StepRecord
            {
                Step = _step,
                Time = (float)(_step * (double)_dt),
                Action = action,
                State = state,
                EpisodeEnd = terminal,
            };
            _pendingCount++;
            _framesCaptured++;
            _step++;

            if (terminal)
            {
                CloseEpisode(_episode, _step);
                _phase = Phase.Ending;
                SetPause(PauseReason.Episode, true);
            }
        }

        void CloseEpisode(EpisodeInfo episode, int numSteps)
        {
            episode.NumSteps = numSteps;
            var settings = new Dictionary<string, object>();
            try
            {
                _sourceHooks?.DescribeSettings(settings);
            }
            catch (Exception e)
            {
                Debug.LogException(e, this);
            }
            episode.ActionSourceSettings = settings;
            _episodesCompleted++;
        }

        // Hands finished readbacks to the writer, oldest first, so frames stay in step order.
        void DrainReadbacks(bool flushAll)
        {
            while (_pendingCount > 0)
            {
                ref PendingFrame f = ref _pending[_pendingHead];
                if (!f.Request.done)
                {
                    if (flushAll)
                    {
                        f.Request.WaitForCompletion();
                    }
                    else
                    {
                        f.Request.Update();
                        if (!f.Request.done) return;
                    }
                }

                if (_pendingCount == 1 && !f.Step.EpisodeEnd)
                {
                    // The newest step is held back so an abrupt stop can still mark it as the
                    // episode's final step (exactly one episode_end line, and it is the last).
                    if (!flushAll) return;
                    f.Step.EpisodeEnd = true;
                    CloseEpisode(f.Episode, f.Step.Step + 1);
                }

                Submit(ref f);
                if (f.Step.EpisodeEnd) _writer.EnqueueEpisodeEnd(f.Episode);
                f = default;
                _pendingHead = (_pendingHead + 1) % _pending.Length;
                _pendingCount--;
            }
        }

        void Submit(ref PendingFrame f)
        {
            if (f.Request.hasError)
            {
                _readbackErrors++;
                f.Episode.MarkCorrupted("GPU readback failed at step " + f.Step.Step + ".");
                _writer.EnqueueStep(f.Episode, f.Step, null);
                return;
            }

            NativeArray<byte> data = f.Request.GetData<byte>();
            byte[] buffer = _pool.Rent();
            if (data.Length != buffer.Length)
            {
                _readbackErrors++;
                f.Episode.MarkCorrupted("GPU readback had an unexpected size at step " + f.Step.Step + ".");
                _pool.Return(buffer);
                _writer.EnqueueStep(f.Episode, f.Step, null);
                return;
            }
            data.CopyTo(buffer);
            _writer.EnqueueStep(f.Episode, f.Step, buffer);
        }

        // Pausing at 75% and resuming at 50% avoids toggling every frame near the limit.
        // If the queue still fills, EnqueueStep blocks, so no frame is ever dropped.
        void UpdateBackpressure()
        {
            int depth = _writer.QueueDepth;
            int capacity = _writer.Capacity;
            if (depth * 4 >= capacity * 3) SetPause(PauseReason.Backpressure, true);
            else if (depth * 2 <= capacity) SetPause(PauseReason.Backpressure, false);
        }

        // Game logic is paused through Time.timeScale, which stops FixedUpdate for every script
        // while Update keeps running, so readbacks and the writer can catch up.
        void SetPause(PauseReason reason, bool on)
        {
            PauseReason before = _pause;
            _pause = on ? _pause | reason : _pause & ~reason;
            if (before == PauseReason.None && _pause != PauseReason.None)
            {
                _savedTimeScale = Time.timeScale;
                Time.timeScale = 0f;
            }
            else if (before != PauseReason.None && _pause == PauseReason.None)
            {
                Time.timeScale = _savedTimeScale;
            }
        }

        bool SessionShouldEnd()
        {
            if (_stopAfterEpisode || _replay != null) return true;
            int maxEpisodes = MaxEpisodesOverride > 0 ? MaxEpisodesOverride : config.maxEpisodes;
            if (maxEpisodes > 0 && _episodesCompleted >= maxEpisodes) return true;
            return config.sessionMinutes > 0f &&
                   Time.realtimeSinceStartup - _sessionStartRealtime >= config.sessionMinutes * 60f;
        }

        void Fail(string message)
        {
            _lastError = message;
            Debug.LogError("[LUCID] " + message, this);
        }
    }
}
