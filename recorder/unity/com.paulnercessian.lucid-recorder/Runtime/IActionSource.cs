namespace Lucid.Recorder
{
    /// <summary>Where each step's action comes from: a human, an autopilot or a replay.</summary>
    public interface IActionSource
    {
        // Called once per logic step, before the game applies the action.
        UnifiedAction GetAction(GameStateSnapshot state);
        string Name { get; }
    }
}
