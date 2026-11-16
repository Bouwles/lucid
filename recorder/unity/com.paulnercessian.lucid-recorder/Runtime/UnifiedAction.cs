using System;

namespace Lucid.Recorder
{
    /// <summary>
    /// One logic step's input in the shared action format (docs/formats.md, "Unified action").
    /// </summary>
    [Serializable]
    public struct UnifiedAction : IEquatable<UnifiedAction>
    {
        public byte Buttons;   // bit flags: left, right, up, down, jump, a, b, reserved
        public float LookX;    // [-1, 1], 0 for 2D games
        public float LookY;    // [-1, 1]

        public UnifiedAction(byte buttons, float lookX = 0f, float lookY = 0f)
        {
            Buttons = buttons;
            LookX = lookX;
            LookY = lookY;
        }

        public bool IsHeld(byte button) => (Buttons & button) != 0;

        /// <summary>
        /// Cancels opposing directions and clamps look to [-1, 1]. NaN look becomes 0.
        /// The recorder applies this to every action before it reaches the game or the file.
        /// </summary>
        public UnifiedAction Sanitized()
        {
            return new UnifiedAction(LucidButtons.CancelOpposites(Buttons), ClampLook(LookX), ClampLook(LookY));
        }

        static float ClampLook(float v)
        {
            if (float.IsNaN(v)) return 0f;
            return v < -1f ? -1f : (v > 1f ? 1f : v);
        }

        public bool Equals(UnifiedAction other) =>
            Buttons == other.Buttons && LookX.Equals(other.LookX) && LookY.Equals(other.LookY);

        public override bool Equals(object obj) => obj is UnifiedAction other && Equals(other);

        public override int GetHashCode()
        {
            unchecked
            {
                int h = Buttons;
                h = h * 397 ^ LookX.GetHashCode();
                h = h * 397 ^ LookY.GetHashCode();
                return h;
            }
        }

        public override string ToString() => $"buttons={Buttons} look=({LookX}, {LookY})";
    }

    /// <summary>Bit values of <see cref="UnifiedAction.Buttons"/>.</summary>
    public static class LucidButtons
    {
        public const byte Left = 1 << 0;
        public const byte Right = 1 << 1;
        public const byte Up = 1 << 2;
        public const byte Down = 1 << 3;
        public const byte Jump = 1 << 4;
        public const byte ActionA = 1 << 5;
        public const byte ActionB = 1 << 6;
        public const byte Reserved = 1 << 7;

        /// <summary>Names in bit order, as used in docs/formats.md.</summary>
        public static readonly string[] Names =
            { "left", "right", "up", "down", "jump", "action_a", "action_b", "reserved" };

        /// <summary>Left and right together become neither; same for up and down.</summary>
        public static byte CancelOpposites(byte buttons)
        {
            const int leftRight = Left | Right;
            const int upDown = Up | Down;
            int b = buttons;
            if ((b & leftRight) == leftRight) b &= ~leftRight;
            if ((b & upDown) == upDown) b &= ~upDown;
            return (byte)b;
        }
    }
}
