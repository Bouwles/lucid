#if LUCID_RECORDING
using System.Collections.Generic;
using UnityEngine;

namespace Lucid.Recorder.Samples
{
    /// <summary>
    /// Lane-runner autopilot. AutoPilotBase handles the behaviour mix (random, competent, noisy,
    /// idle); this class only says what a competent player would press, using the adapter's
    /// obstacle look-ahead.
    /// </summary>
    public sealed class RunnerAutoPilot : AutoPilotBase
    {
        [SerializeField] RunnerAdapter adapter;
        [SerializeField, Min(1)] int laneCount = 3;

        [Tooltip("How far ahead to look for obstacles, in world units.")]
        [SerializeField, Min(0f)] float lookAheadDistance = 15f;

        [Tooltip("React when the nearest obstacle is this many seconds away at the current speed.")]
        [SerializeField, Min(0f)] float reactSeconds = 0.5f;

        [Tooltip("Steps with no buttons after each tap, so the adapter sees a new key press next time.")]
        [SerializeField, Min(1)] int releaseSteps = 2;

        int _release;

        protected override byte UsedButtons =>
            LucidButtons.Left | LucidButtons.Right | LucidButtons.Jump | LucidButtons.Down;

        protected override void OnEpisodeStart(int seed) => _release = 0;

        protected override UnifiedAction DecideAvoidance(GameStateSnapshot state)
        {
            if (_release > 0)
            {
                _release--;
                return default;
            }
            if (adapter == null || !state.Alive || !state.Lane.HasValue) return default;

            int lane = state.Lane.Value;
            if (!adapter.TryGetObstacleAhead(lane, lookAheadDistance, out float distance, out ObstacleKind kind))
                return default;
            if (distance / Mathf.Max(state.Speed, 0.01f) > reactSeconds) return default;

            bool grounded = state.Vertical == VerticalState.Grounded;
            switch (kind)
            {
                case ObstacleKind.Low:
                    return grounded ? Tap(LucidButtons.Jump) : default;
                case ObstacleKind.High:
                    return grounded ? Tap(LucidButtons.Down) : default;
                default:
                    int target = PickFreeLane(lane);
                    if (target < lane) return Tap(LucidButtons.Left);
                    if (target > lane) return Tap(LucidButtons.Right);
                    // Boxed in on both sides: jumping is the only chance left.
                    return grounded ? Tap(LucidButtons.Jump) : default;
            }
        }

        int PickFreeLane(int lane)
        {
            bool leftFree = lane > 0 && !Blocked(lane - 1);
            bool rightFree = lane < laneCount - 1 && !Blocked(lane + 1);
            if (leftFree && rightFree) return Rng.Next(2) == 0 ? lane - 1 : lane + 1;
            if (leftFree) return lane - 1;
            if (rightFree) return lane + 1;
            return lane;
        }

        bool Blocked(int lane) => adapter.TryGetObstacleAhead(lane, lookAheadDistance, out _, out _);

        UnifiedAction Tap(byte button)
        {
            _release = releaseSteps;
            return new UnifiedAction(button);
        }

        protected override void OnDescribeSettings(IDictionary<string, object> settings)
        {
            settings["look_ahead_distance"] = lookAheadDistance;
            settings["react_seconds"] = reactSeconds;
        }
    }
}
#endif
