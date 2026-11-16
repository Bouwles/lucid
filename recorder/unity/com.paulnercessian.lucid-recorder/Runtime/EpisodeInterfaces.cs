using System.Collections.Generic;

namespace Lucid.Recorder
{
    /// <summary>
    /// Implemented by the game adapter next to <see cref="IGameAdapter"/>, so the recorder can
    /// restart the game for each episode with a known seed.
    /// </summary>
    public interface IEpisodeControl
    {
        /// <summary>
        /// Seed the game's single gameplay RNG and reset to the start of a run. Called from Update
        /// while game logic is paused. The game must not restart itself on death while recording.
        /// </summary>
        void BeginEpisode(int seed);

        /// <summary>
        /// True once the new run is fully set up and on screen. The recorder waits at least one
        /// render frame and until this is true before recording step 0.
        /// </summary>
        bool EpisodeReady { get; }
    }

    /// <summary>Optional for action sources: episode start notification and settings for meta.json.</summary>
    public interface IEpisodeAwareSource
    {
        void BeginEpisode(int seed);

        /// <summary>Fills action_source.settings in meta.json. Called when the episode ends.</summary>
        void DescribeSettings(IDictionary<string, object> settings);
    }
}
