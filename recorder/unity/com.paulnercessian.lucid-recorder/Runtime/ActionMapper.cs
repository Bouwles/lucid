using System;
using System.Collections.Generic;
using UnityEngine;

namespace Lucid.Recorder
{
    /// <summary>
    /// Per-game mapping from the game's own control names ("slide", "fire") to unified button bits,
    /// plus the look scale for games with mouse look. One asset per game; it should match
    /// docs/games/&lt;game_id&gt;.md.
    /// </summary>
    [CreateAssetMenu(menuName = "LUCID/Action Mapper", fileName = "ActionMapper")]
    public sealed class ActionMapper : ScriptableObject
    {
        public enum UnifiedButton
        {
            Left = 0,
            Right = 1,
            Up = 2,
            Down = 3,
            Jump = 4,
            ActionA = 5,
            ActionB = 6,
        }

        [Serializable]
        public struct Binding
        {
            public string control;
            public UnifiedButton button;
        }

        [SerializeField]
        Binding[] bindings =
        {
            new Binding { control = "left", button = UnifiedButton.Left },
            new Binding { control = "right", button = UnifiedButton.Right },
            new Binding { control = "jump", button = UnifiedButton.Jump },
            new Binding { control = "slide", button = UnifiedButton.Down },
        };

        [Tooltip("Off for 2D games: look is always [0, 0] and meta.json gets look_scale null.")]
        [SerializeField] bool usesLook;

        [Tooltip("Raw look delta per step that maps to 1.0. Larger deltas are clipped. Calibrate per input backend, " +
                 "since the legacy Input Manager and the Input System report mouse movement in different units.")]
        [SerializeField, Min(0.0001f)] float lookScale = 20f;

        [SerializeField] bool invertLookY;

        Dictionary<string, byte> _bits;

        public bool UsesLook => usesLook;
        public float LookScale => lookScale;

        /// <summary>The value for look_scale in meta.json.</summary>
        public float? LookScaleForMeta => usesLook ? lookScale : (float?)null;

        /// <summary>Unified bit(s) for a game control name, or 0 if the control is not mapped.</summary>
        public byte BitFor(string control)
        {
            EnsureMap();
            return control != null && _bits.TryGetValue(control, out byte bit) ? bit : (byte)0;
        }

        public bool IsHeld(in UnifiedAction action, string control) => (action.Buttons & BitFor(control)) != 0;

        public byte Press(byte buttons, string control) => (byte)(buttons | BitFor(control));

        /// <summary>Clips a raw look delta to +-lookScale and divides by it, giving [-1, 1].</summary>
        public Vector2 MapLook(float rawX, float rawY)
        {
            if (!usesLook) return Vector2.zero;
            float x = Mathf.Clamp(rawX / lookScale, -1f, 1f);
            float y = Mathf.Clamp(rawY / lookScale, -1f, 1f);
            return new Vector2(x, invertLookY ? -y : y);
        }

        void OnEnable() => _bits = null;

        void OnValidate() => _bits = null;

        void EnsureMap()
        {
            if (_bits != null) return;
            _bits = new Dictionary<string, byte>(StringComparer.Ordinal);
            if (bindings == null) return;
            foreach (Binding b in bindings)
            {
                if (string.IsNullOrEmpty(b.control)) continue;
                _bits.TryGetValue(b.control, out byte existing);
                _bits[b.control] = (byte)(existing | (1 << (int)b.button));
            }
        }
    }
}
