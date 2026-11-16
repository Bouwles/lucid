#if ENABLE_INPUT_SYSTEM && LUCID_INPUT_SYSTEM
#define LUCID_NEW_INPUT
#endif

using System.Collections.Generic;
using UnityEngine;
#if LUCID_NEW_INPUT
using UnityEngine.InputSystem;
#endif

namespace Lucid.Recorder
{
    /// <summary>
    /// Keyboard (and optional mouse look) input from a person playing.
    /// Arrows or WASD = directions, Space = jump, Z or J = action_a, X or K = action_b.
    /// Priority rule: every held key sets its bit (buttons are multi-hot), then opposite
    /// directions cancel. A key pressed at any time since the previous logic step counts as held
    /// for that step, so taps shorter than one step are not lost.
    /// Uses the Input System package when it is installed and enabled, else the legacy Input Manager.
    /// </summary>
    [AddComponentMenu("LUCID/Human Action Source")]
    public sealed class HumanActionSource : MonoBehaviour, IActionSource, IEpisodeAwareSource
    {
        [Tooltip("Supplies the look scale. The recorder replaces it with its config's mapper if that is set.")]
        [SerializeField] ActionMapper mapper;
        [SerializeField] bool useMouseLook;
        [SerializeField] bool lockCursor = true;

        byte _latched;
        float _lookX;
        float _lookY;

        public string Name => "human";

        public ActionMapper Mapper
        {
            get => mapper;
            set => mapper = value;
        }

        void Update()
        {
            _latched |= ReadButtons();
            if (useMouseLook)
            {
                Vector2 d = ReadMouseDelta();
                _lookX += d.x;
                _lookY += d.y;
            }
        }

        public UnifiedAction GetAction(GameStateSnapshot state)
        {
            byte buttons = (byte)(_latched | ReadButtons());
            _latched = 0;
            Vector2 look = useMouseLook && mapper != null ? mapper.MapLook(_lookX, _lookY) : Vector2.zero;
            _lookX = 0f;
            _lookY = 0f;
            return new UnifiedAction(LucidButtons.CancelOpposites(buttons), look.x, look.y);
        }

        public void BeginEpisode(int seed)
        {
            _latched = 0;
            _lookX = 0f;
            _lookY = 0f;
            if (useMouseLook && lockCursor) Cursor.lockState = CursorLockMode.Locked;
        }

        public void DescribeSettings(IDictionary<string, object> settings)
        {
            settings["mouse_look"] = useMouseLook;
#if LUCID_NEW_INPUT
            settings["input_backend"] = "input_system";
#else
            settings["input_backend"] = "input_manager";
#endif
        }

        void OnDisable()
        {
            if (useMouseLook && lockCursor) Cursor.lockState = CursorLockMode.None;
        }

#if LUCID_NEW_INPUT
        static byte ReadButtons()
        {
            Keyboard kb = Keyboard.current;
            if (kb == null) return 0;
            int b = 0;
            if (kb.leftArrowKey.isPressed || kb.aKey.isPressed) b |= LucidButtons.Left;
            if (kb.rightArrowKey.isPressed || kb.dKey.isPressed) b |= LucidButtons.Right;
            if (kb.upArrowKey.isPressed || kb.wKey.isPressed) b |= LucidButtons.Up;
            if (kb.downArrowKey.isPressed || kb.sKey.isPressed) b |= LucidButtons.Down;
            if (kb.spaceKey.isPressed) b |= LucidButtons.Jump;
            if (kb.zKey.isPressed || kb.jKey.isPressed) b |= LucidButtons.ActionA;
            if (kb.xKey.isPressed || kb.kKey.isPressed) b |= LucidButtons.ActionB;
            return (byte)b;
        }

        static Vector2 ReadMouseDelta()
        {
            Mouse mouse = Mouse.current;
            return mouse != null ? mouse.delta.ReadValue() : Vector2.zero;
        }
#elif ENABLE_LEGACY_INPUT_MANAGER
        static byte ReadButtons()
        {
            int b = 0;
            if (Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.A)) b |= LucidButtons.Left;
            if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D)) b |= LucidButtons.Right;
            if (Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.W)) b |= LucidButtons.Up;
            if (Input.GetKey(KeyCode.DownArrow) || Input.GetKey(KeyCode.S)) b |= LucidButtons.Down;
            if (Input.GetKey(KeyCode.Space)) b |= LucidButtons.Jump;
            if (Input.GetKey(KeyCode.Z) || Input.GetKey(KeyCode.J)) b |= LucidButtons.ActionA;
            if (Input.GetKey(KeyCode.X) || Input.GetKey(KeyCode.K)) b |= LucidButtons.ActionB;
            return (byte)b;
        }

        static Vector2 ReadMouseDelta() => new Vector2(Input.GetAxisRaw("Mouse X"), Input.GetAxisRaw("Mouse Y"));
#else
        static byte ReadButtons() => 0;

        static Vector2 ReadMouseDelta() => Vector2.zero;
#endif
    }
}
