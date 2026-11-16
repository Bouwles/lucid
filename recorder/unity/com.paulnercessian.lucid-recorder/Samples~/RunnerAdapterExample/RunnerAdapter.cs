#if LUCID_RECORDING
using UnityEngine;

namespace Lucid.Recorder.Samples
{
    /// <summary>
    /// Connects a lane runner (left, right, jump, slide) to the LUCID recorder.
    /// Put it in the recording scene only. Every TODO marks a line that has to call your game code.
    /// In recording mode the game must read its input only from Apply(), never from the keyboard,
    /// otherwise replays and the determinism test cannot reproduce a run.
    /// </summary>
    public sealed class RunnerAdapter : MonoBehaviour, IGameAdapter, IEpisodeControl
    {
        [Tooltip("Must match the id in configs/games.yaml.")]
        [SerializeField] string gameId = "runner";

        [Tooltip("Maps the controls left, right, jump and slide to unified buttons. Empty uses left, right, jump, down.")]
        [SerializeField] ActionMapper mapper;

        byte _leftBit;
        byte _rightBit;
        byte _jumpBit;
        byte _slideBit;
        byte _previous;
        bool _ready;

        public string GameId => gameId;
        public bool EpisodeReady => _ready;

        void Awake()
        {
            _leftBit = Bit("left", LucidButtons.Left);
            _rightBit = Bit("right", LucidButtons.Right);
            _jumpBit = Bit("jump", LucidButtons.Jump);
            _slideBit = Bit("slide", LucidButtons.Down);
        }

        byte Bit(string control, byte fallback)
        {
            byte bit = mapper != null ? mapper.BitFor(control) : (byte)0;
            return bit != 0 ? bit : fallback;
        }

        public void BeginEpisode(int seed)
        {
            _previous = 0;
            _ready = false;

            // TODO: seed the game's single gameplay RNG, for example GameRandom.Init(seed).
            //       Every random choice in gameplay (obstacle spawns, coin patterns) must use it.
            // TODO: reset the run to its start state: player in the start lane, speed and distance
            //       reset, obstacles cleared, camera snapped to its start position.
            //       If the reset reloads a scene, set _ready = true from SceneManager.sceneLoaded instead.
            // TODO: make sure the game does not restart by itself after death while recording.
            //       The recorder holds the game-over frame for its configured time, then calls this.

            _ready = true;
        }

        public void Apply(UnifiedAction action)
        {
            // The runner reacts to key presses, so only rising edges trigger a move.
            byte pressed = (byte)(action.Buttons & ~_previous);
            _previous = action.Buttons;

            if ((pressed & _leftBit) != 0)
            {
                // TODO: player.MoveLeft();
            }
            if ((pressed & _rightBit) != 0)
            {
                // TODO: player.MoveRight();
            }
            if ((pressed & _jumpBit) != 0)
            {
                // TODO: player.Jump();
            }
            if ((pressed & _slideBit) != 0)
            {
                // TODO: player.Slide();
                //       If slide lasts while the key is held, use (action.Buttons & _slideBit) != 0 instead.
            }
        }

        public GameStateSnapshot ReadState()
        {
            // TODO: replace each value with a read from your game. This must not change game state.
            return new GameStateSnapshot
            {
                Lane = 1,                          // TODO: player.Lane, 0 = leftmost
                X = 0f,                            // TODO: player.transform.position.x
                Y = 0f,                            // TODO: player.transform.position.y
                Vertical = VerticalState.Grounded, // TODO: Jumping, Sliding or Falling from the player's state
                Alive = true,                      // TODO: !player.IsDead
                Speed = 0f,                        // TODO: game.Speed (world units per second)
                Distance = 0f,                     // TODO: game.Distance
            };
        }

        /// <summary>Look-ahead query for RunnerAutoPilot: the nearest obstacle ahead in a lane.</summary>
        public bool TryGetObstacleAhead(int lane, float maxDistance, out float distance, out ObstacleKind kind)
        {
            distance = 0f;
            kind = ObstacleKind.Full;

            // TODO: search the active obstacles, for example:
            // float best = float.MaxValue;
            // foreach (Obstacle o in spawner.Active)
            // {
            //     float ahead = o.transform.position.z - player.transform.position.z;
            //     if (o.Lane == lane && ahead > 0f && ahead <= maxDistance && ahead < best)
            //     {
            //         best = ahead;
            //         kind = o.CanJumpOver ? ObstacleKind.Low : o.CanSlideUnder ? ObstacleKind.High : ObstacleKind.Full;
            //     }
            // }
            // distance = best;
            // return best < float.MaxValue;

            return false;
        }
    }

    /// <summary>Low: jump over it. High: slide under it. Full: change lane.</summary>
    public enum ObstacleKind
    {
        Low,
        High,
        Full,
    }
}
#endif
