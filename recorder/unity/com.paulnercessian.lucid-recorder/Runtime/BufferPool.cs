using System.Collections.Generic;

namespace Lucid.Recorder
{
    /// <summary>
    /// Fixed-size byte buffers shared by the game thread (fills them from GPU readbacks) and the
    /// writer thread (returns them after encoding). Preallocated so recording does not allocate
    /// per frame; it only grows if more buffers are in flight than expected.
    /// </summary>
    internal sealed class BufferPool
    {
        readonly Stack<byte[]> _free;
        readonly object _lock = new object();

        public int BufferSize { get; }

        public BufferPool(int bufferSize, int preallocate)
        {
            BufferSize = bufferSize;
            _free = new Stack<byte[]>(preallocate);
            for (int i = 0; i < preallocate; i++) _free.Push(new byte[bufferSize]);
        }

        public byte[] Rent()
        {
            lock (_lock)
            {
                if (_free.Count > 0) return _free.Pop();
            }
            return new byte[BufferSize];
        }

        public void Return(byte[] buffer)
        {
            if (buffer == null || buffer.Length != BufferSize) return;
            lock (_lock) _free.Push(buffer);
        }
    }
}
