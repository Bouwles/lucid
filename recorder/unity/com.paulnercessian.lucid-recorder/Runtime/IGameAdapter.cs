namespace Lucid.Recorder
{
    public interface IGameAdapter
    {
        // Implemented once per game: applies a UnifiedAction to that game's controls,
        // and reports the label state (Section 6.3).
        void Apply(UnifiedAction action);
        GameStateSnapshot ReadState();
        string GameId { get; }
    }
}
