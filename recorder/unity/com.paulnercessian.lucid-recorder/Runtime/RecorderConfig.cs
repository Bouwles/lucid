using System;
using System.IO;
using UnityEngine;

namespace Lucid.Recorder
{
    /// <summary>All recorder settings for one game. Create via Assets > Create > LUCID > Recorder Config.</summary>
    [CreateAssetMenu(menuName = "LUCID/Recorder Config", fileName = "LucidRecorderConfig")]
    public sealed class RecorderConfig : ScriptableObject
    {
        /// <summary>Auto detects at startup how ImageConversion orders rows, so row 0 of every PNG is the top.</summary>
        public enum RowOrder
        {
            Auto,
            Flip,
            NoFlip,
        }

        [Header("Output")]
        [Tooltip("Folder that receives <session_id>/ folders, usually <LUCID repo>/data/raw. " +
                 "Relative paths are relative to the Unity project (editor) or the build folder. " +
                 "Empty means <persistentDataPath>/lucid_raw.")]
        public string outputFolder = "";

        [Tooltip("game_build_version in the metadata. Empty uses Application.version.")]
        public string gameBuildVersion = "";

        [Header("Frames")]
        [Tooltip("Recording width in pixels. Keep the game's aspect ratio, short side 256 or more.")]
        [Min(16)] public int width = 256;
        [Min(16)] public int height = 256;

        [Tooltip("Layers the recording camera never draws (UI by default).")]
        public LayerMask excludedLayers = 1 << 5;

        [Tooltip("Leave on Auto unless frames come out upside down.")]
        public RowOrder rowOrder = RowOrder.Auto;

        [Header("Session")]
        [Tooltip("Stop after the first episode that ends past this many minutes of wall-clock time. 0 = no limit.")]
        [Min(0f)] public float sessionMinutes = 30f;

        [Tooltip("Stop after this many episodes. 0 = no limit.")]
        [Min(0)] public int maxEpisodes = 0;

        [Tooltip("End an episode after this many seconds of game time even if the player is alive. 0 = no limit.")]
        [Min(0f)] public float maxEpisodeSeconds = 180f;

        [Tooltip("Seed for the sequence of episode seeds. 0 = derived from the clock.")]
        public int sessionSeed = 0;

        [Tooltip("How long the game-over frame is recorded after death (alive false, buttons 0).")]
        [Min(0f)] public float gameOverHoldSeconds = 1f;

        [Header("Writer")]
        [Tooltip("Frames that may wait for the writer thread. Game logic pauses at 75% full and resumes at 50%.")]
        [Min(4)] public int queueCapacity = 240;

        [Tooltip("GPU readbacks allowed in flight before the recorder waits for the oldest one.")]
        [Min(1)] public int maxPendingReadbacks = 6;

        [Header("Actions")]
        [Tooltip("The game's control map. Its look scale is written to meta.json.")]
        public ActionMapper actionMapper;

        public AutoPilotSettings autopilot = new AutoPilotSettings();

        /// <summary>Absolute output folder.</summary>
        public string ResolveOutputFolder()
        {
            if (string.IsNullOrWhiteSpace(outputFolder))
                return Path.Combine(Application.persistentDataPath, "lucid_raw");
            if (Path.IsPathRooted(outputFolder)) return Path.GetFullPath(outputFolder);
            string baseDir = Directory.GetParent(Application.dataPath)?.FullName ?? Application.dataPath;
            return Path.GetFullPath(Path.Combine(baseDir, outputFolder));
        }
    }

    /// <summary>Autopilot mix (SPEC 10.5). Saved into each episode's meta.json.</summary>
    [Serializable]
    public sealed class AutoPilotSettings
    {
        [Tooltip("Relative weights for the behaviour of each episode or segment. They need not sum to 1.")]
        [Range(0f, 1f)] public float pRandom = 0.25f;
        [Range(0f, 1f)] public float pCompetent = 0.35f;
        [Range(0f, 1f)] public float pNoisy = 0.3f;
        [Range(0f, 1f)] public float pIdle = 0.1f;

        [Tooltip("Random-with-persistence: how long each random action is held.")]
        [Min(0f)] public float holdMinSeconds = 0.1f;
        [Min(0f)] public float holdMaxSeconds = 0.8f;

        [Tooltip("Length of an idle stretch when behaviours switch within an episode.")]
        [Min(0f)] public float idleMinSeconds = 0.5f;
        [Min(0f)] public float idleMaxSeconds = 2f;

        [Tooltip("Noisy avoider: chance that an avoiding action becomes a wrong or late one.")]
        [Range(0f, 1f)] public float mistakeProbability = 0.15f;

        [Tooltip("Noisy avoider: how long a late reaction waits.")]
        [Min(0f)] public float lateMinSeconds = 0.1f;
        [Min(0f)] public float lateMaxSeconds = 0.4f;

        [Tooltip("Draw a new behaviour every few seconds instead of keeping one per episode.")]
        public bool switchWithinEpisode = true;

        [Tooltip("Mean length of a behaviour segment when switching is on.")]
        [Min(0.1f)] public float meanSegmentSeconds = 6f;
    }
}
