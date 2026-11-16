using System.Collections.Generic;
using UnityEngine;

namespace Lucid.Recorder
{
    /// <summary>
    /// Shared autopilot logic (SPEC 10.5). Mixes four behaviours, chosen per episode or switched
    /// within an episode. A game subclass supplies the look-ahead (<see cref="DecideAvoidance"/>)
    /// and the buttons it uses. All randomness comes from <see cref="Rng"/>, which is seeded from the
    /// episode seed, so a session with a fixed session seed is reproducible.
    /// </summary>
    public abstract class AutoPilotBase : MonoBehaviour, IActionSource, IEpisodeAwareSource
    {
        public enum Behaviour
        {
            RandomPersistent,
            CompetentAvoider,
            NoisyAvoider,
            Idle,
        }

        const int MaxLoggedBehaviours = 64;

        [Tooltip("Written to meta.json as action_source.name.")]
        [SerializeField] string sourceName = "autopilot";

        [Tooltip("Used when the autopilot runs without a recorder. The recorder replaces these with its config's settings.")]
        [SerializeField] AutoPilotSettings settings = new AutoPilotSettings();

        System.Random _rng = new System.Random(0);
        float _dt = 0.02f;
        Behaviour _current;
        int _segmentStepsLeft;
        UnifiedAction _held;
        int _holdStepsLeft;
        UnifiedAction _mistake;
        int _mistakeStepsLeft;
        int _lateStepsLeft;
        int _switches;
        byte[] _choices;
        readonly List<string> _log = new List<string>(MaxLoggedBehaviours);

        public virtual string Name => string.IsNullOrEmpty(sourceName) ? "autopilot" : sourceName;
        public Behaviour CurrentBehaviour => _current;
        public AutoPilotSettings Settings => settings;

        /// <summary>The autopilot's own RNG. Never use it for game logic.</summary>
        protected System.Random Rng => _rng;

        /// <summary>Length of one logic step in seconds.</summary>
        protected float StepSeconds => _dt;

        /// <summary>Buttons this game uses; random actions are drawn from these.</summary>
        protected abstract byte UsedButtons { get; }

        /// <summary>
        /// The look-ahead hook: what a competent player would press this step, given the state and
        /// whatever the subclass reads from the game (upcoming obstacles, gaps, enemies).
        /// Return default for no action.
        /// </summary>
        protected abstract UnifiedAction DecideAvoidance(GameStateSnapshot state);

        /// <summary>Button combinations the random behaviour picks from. Default: nothing, or one used button.</summary>
        protected virtual byte[] RandomChoices()
        {
            var list = new List<byte> { 0 };
            for (int bit = 0; bit < 7; bit++)
            {
                byte b = (byte)(1 << bit);
                if ((UsedButtons & b) != 0) list.Add(b);
            }
            return list.ToArray();
        }

        /// <summary>Look vector for a random action. Default zero (2D games).</summary>
        protected virtual Vector2 SampleRandomLook() => Vector2.zero;

        protected virtual void OnEpisodeStart(int seed) { }

        /// <summary>Called every step before the behaviour runs, for subclasses that track state.</summary>
        protected virtual void Observe(GameStateSnapshot state) { }

        protected virtual void OnDescribeSettings(IDictionary<string, object> settings) { }

        public void Configure(AutoPilotSettings newSettings)
        {
            if (newSettings != null) settings = newSettings;
        }

        public void BeginEpisode(int seed)
        {
            // XOR with a constant so the autopilot's stream differs from the game's RNG seeded with the same value.
            _rng = new System.Random(seed ^ 0x2545F491);
            _dt = Time.fixedDeltaTime > 0f ? Time.fixedDeltaTime : 0.02f;
            _choices = RandomChoices();
            _held = default;
            _holdStepsLeft = 0;
            _mistakeStepsLeft = 0;
            _lateStepsLeft = 0;
            _switches = 0;
            _log.Clear();
            OnEpisodeStart(seed);
            Enter(Draw());
        }

        public UnifiedAction GetAction(GameStateSnapshot state)
        {
            if (_choices == null) BeginEpisode(0);
            Observe(state);
            if (settings.switchWithinEpisode && --_segmentStepsLeft <= 0)
            {
                Enter(Draw());
                _switches++;
            }

            switch (_current)
            {
                case Behaviour.RandomPersistent: return RandomStep();
                case Behaviour.CompetentAvoider: return DecideAvoidance(state);
                case Behaviour.NoisyAvoider: return NoisyStep(state);
                default: return default;
            }
        }

        public void DescribeSettings(IDictionary<string, object> s)
        {
            s["p_random"] = settings.pRandom;
            s["p_competent"] = settings.pCompetent;
            s["p_noisy"] = settings.pNoisy;
            s["p_idle"] = settings.pIdle;
            s["hold_min_s"] = settings.holdMinSeconds;
            s["hold_max_s"] = settings.holdMaxSeconds;
            s["idle_min_s"] = settings.idleMinSeconds;
            s["idle_max_s"] = settings.idleMaxSeconds;
            s["mistake_probability"] = settings.mistakeProbability;
            s["late_min_s"] = settings.lateMinSeconds;
            s["late_max_s"] = settings.lateMaxSeconds;
            s["switch_within_episode"] = settings.switchWithinEpisode;
            s["mean_segment_s"] = settings.meanSegmentSeconds;
            s["behaviours"] = new List<string>(_log);
            s["behaviour_switches"] = _switches;
            OnDescribeSettings(s);
        }

        public static string WireName(Behaviour b)
        {
            switch (b)
            {
                case Behaviour.RandomPersistent: return "random_persistent";
                case Behaviour.CompetentAvoider: return "competent_avoider";
                case Behaviour.NoisyAvoider: return "noisy_avoider";
                default: return "idle";
            }
        }

        Behaviour Draw()
        {
            float total = settings.pRandom + settings.pCompetent + settings.pNoisy + settings.pIdle;
            if (total <= 0f) return Behaviour.Idle;
            double u = _rng.NextDouble() * total;
            if ((u -= settings.pRandom) < 0) return Behaviour.RandomPersistent;
            if ((u -= settings.pCompetent) < 0) return Behaviour.CompetentAvoider;
            if ((u -= settings.pNoisy) < 0) return Behaviour.NoisyAvoider;
            return Behaviour.Idle;
        }

        void Enter(Behaviour b)
        {
            _current = b;
            _holdStepsLeft = 0;
            _mistakeStepsLeft = 0;
            _lateStepsLeft = 0;
            if (_log.Count < MaxLoggedBehaviours) _log.Add(WireName(b));

            if (!settings.switchWithinEpisode) _segmentStepsLeft = int.MaxValue;
            else if (b == Behaviour.Idle) _segmentStepsLeft = StepsBetween(settings.idleMinSeconds, settings.idleMaxSeconds);
            else
            {
                // Exponential segment lengths: switches happen at a steady rate with no fixed rhythm.
                double seconds = -System.Math.Log(1.0 - _rng.NextDouble()) * settings.meanSegmentSeconds;
                _segmentStepsLeft = Mathf.Max(1, (int)System.Math.Round(seconds / _dt));
            }
        }

        UnifiedAction RandomStep()
        {
            if (_holdStepsLeft <= 0)
            {
                Vector2 look = SampleRandomLook();
                _held = new UnifiedAction(_choices[_rng.Next(_choices.Length)], look.x, look.y);
                _holdStepsLeft = StepsBetween(settings.holdMinSeconds, settings.holdMaxSeconds);
            }
            _holdStepsLeft--;
            return _held;
        }

        UnifiedAction NoisyStep(GameStateSnapshot state)
        {
            if (_mistakeStepsLeft > 0)
            {
                _mistakeStepsLeft--;
                return _mistake;
            }
            if (_lateStepsLeft > 0)
            {
                _lateStepsLeft--;
                return default;
            }

            UnifiedAction good = DecideAvoidance(state);
            if (good.Buttons == 0 || _rng.NextDouble() >= settings.mistakeProbability) return good;

            if (_rng.NextDouble() < 0.5)
            {
                _mistake = new UnifiedAction(WrongChoice(good.Buttons));
                _mistakeStepsLeft = StepsBetween(settings.holdMinSeconds, settings.holdMaxSeconds) - 1;
                return _mistake;
            }
            _lateStepsLeft = StepsBetween(settings.lateMinSeconds, settings.lateMaxSeconds) - 1;
            return default;
        }

        byte WrongChoice(byte correct)
        {
            if (_choices.Length <= 1) return 0;
            for (int attempt = 0; attempt < 8; attempt++)
            {
                byte b = _choices[_rng.Next(_choices.Length)];
                if (b != correct) return b;
            }
            return 0;
        }

        int StepsBetween(float minSeconds, float maxSeconds)
        {
            int lo = Mathf.Max(1, Mathf.RoundToInt(minSeconds / _dt));
            int hi = Mathf.Max(lo, Mathf.RoundToInt(maxSeconds / _dt));
            return _rng.Next(lo, hi + 1);
        }
    }
}
