using System;

namespace Lucid.Recorder
{
    /// <summary>Vertical state label. Written as the lowercase strings in docs/formats.md.</summary>
    public enum VerticalState
    {
        Grounded,
        Jumping,
        Sliding,
        Falling,
        None,
    }

    /// <summary>
    /// Label state for one logic step. Used for evaluation probes and dataset
    /// statistics only; it is never a model input.
    /// </summary>
    [Serializable]
    public struct GameStateSnapshot
    {
        /// <summary>Lane index, or null for games without lanes.</summary>
        public int? Lane;
        public float X;
        public float Y;
        public VerticalState Vertical;
        public bool Alive;
        public float Speed;
        public float Distance;

        public static string ToWire(VerticalState state)
        {
            switch (state)
            {
                case VerticalState.Grounded: return "grounded";
                case VerticalState.Jumping: return "jumping";
                case VerticalState.Sliding: return "sliding";
                case VerticalState.Falling: return "falling";
                default: return "none";
            }
        }
    }
}
