using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Lucid.Recorder
{
    /// <summary>
    /// Replays the actions of a recorded episode by step index. With the episode's seed (read from
    /// its meta.json) a deterministic game reproduces the original run. Used for determinism tests
    /// and for side-by-side videos of the real game and the dream.
    /// </summary>
    [AddComponentMenu("LUCID/Replay Action Source")]
    public sealed class ReplayActionSource : MonoBehaviour, IActionSource, IEpisodeAwareSource
    {
        [Tooltip("Episode folder (ep_XXXXXX) that holds steps.jsonl and meta.json.")]
        [SerializeField] string episodeFolder = "";

        byte[] _buttons = Array.Empty<byte>();
        float[] _lookX = Array.Empty<float>();
        float[] _lookY = Array.Empty<float>();
        int _count;
        int _index;
        bool _loaded;

        public string Name => "replay";
        public string EpisodeFolder => episodeFolder;
        public int StepCount => _count;
        public bool HasSeed { get; private set; }
        public int Seed { get; private set; }

        /// <summary>True once every recorded action has been handed out.</summary>
        public bool Finished => _index >= _count;

        public void EnsureLoaded()
        {
            if (!_loaded) Load(episodeFolder);
        }

        /// <summary>Reads an episode. Throws if steps.jsonl is missing or malformed.</summary>
        public void Load(string folder)
        {
            if (string.IsNullOrEmpty(folder)) throw new ArgumentException("No episode folder set.");
            string[] lines = File.ReadAllLines(Path.Combine(folder, "steps.jsonl"));
            var buttons = new List<byte>(lines.Length);
            var lookX = new List<float>(lines.Length);
            var lookY = new List<float>(lines.Length);
            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                if (!JsonScan.TryGetLong(line, "step", out long step) || step != buttons.Count)
                    throw new FormatException($"steps.jsonl line {buttons.Count + 1}: expected step {buttons.Count}.");
                if (!JsonScan.TryGetLong(line, "buttons", out long b) || b < 0 || b > 255)
                    throw new FormatException($"steps.jsonl step {step}: bad buttons value.");
                if (!JsonScan.TryGetFloatPair(line, "look", out float lx, out float ly))
                    throw new FormatException($"steps.jsonl step {step}: bad look value.");
                buttons.Add((byte)b);
                lookX.Add(lx);
                lookY.Add(ly);
            }

            HasSeed = false;
            string metaPath = Path.Combine(folder, "meta.json");
            if (File.Exists(metaPath) && JsonScan.TryGetLong(File.ReadAllText(metaPath), "seed", out long seed))
            {
                Seed = (int)seed;
                HasSeed = true;
            }
            else
            {
                Debug.LogWarning($"[LUCID] No seed found in {metaPath}; the replay will use a fresh seed and will not match.");
            }

            episodeFolder = folder;
            _buttons = buttons.ToArray();
            _lookX = lookX.ToArray();
            _lookY = lookY.ToArray();
            _count = _buttons.Length;
            _index = 0;
            _loaded = true;
        }

        public UnifiedAction GetAction(GameStateSnapshot state)
        {
            if (_index >= _count) return default;
            int i = _index++;
            return new UnifiedAction(_buttons[i], _lookX[i], _lookY[i]);
        }

        public void BeginEpisode(int seed)
        {
            EnsureLoaded();
            _index = 0;
        }

        public void DescribeSettings(IDictionary<string, object> settings)
        {
            settings["replayed_episode"] = episodeFolder;
            if (HasSeed) settings["replayed_seed"] = Seed;
        }
    }
}
