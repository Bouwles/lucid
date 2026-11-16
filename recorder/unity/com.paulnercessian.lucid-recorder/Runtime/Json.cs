using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Lucid.Recorder
{
    /// <summary>
    /// Small JSON writer. All numbers use the invariant culture, so a system locale that writes
    /// 0.5 as "0,5" cannot break the files.
    /// </summary>
    public sealed class JsonWriter
    {
        const int MaxDepth = 32;
        static readonly char[] FloatMarks = { '.', 'E', 'e' };

        readonly StringBuilder _sb;
        readonly bool[] _hasItems = new bool[MaxDepth];
        int _depth;
        bool _afterKey;

        public JsonWriter(int capacity = 256)
        {
            _sb = new StringBuilder(capacity);
        }

        public void Clear()
        {
            _sb.Length = 0;
            _depth = 0;
            _afterKey = false;
        }

        public override string ToString() => _sb.ToString();

        public JsonWriter BeginObject()
        {
            BeforeValue();
            _sb.Append('{');
            Push();
            return this;
        }

        public JsonWriter EndObject()
        {
            Pop();
            _sb.Append('}');
            return this;
        }

        public JsonWriter BeginArray()
        {
            BeforeValue();
            _sb.Append('[');
            Push();
            return this;
        }

        public JsonWriter EndArray()
        {
            Pop();
            _sb.Append(']');
            return this;
        }

        public JsonWriter Key(string name)
        {
            Separator();
            AppendString(name);
            _sb.Append(": ");
            _afterKey = true;
            return this;
        }

        public JsonWriter Null()
        {
            BeforeValue();
            _sb.Append("null");
            return this;
        }

        public JsonWriter Value(string v)
        {
            if (v == null) return Null();
            BeforeValue();
            AppendString(v);
            return this;
        }

        public JsonWriter Value(bool v)
        {
            BeforeValue();
            _sb.Append(v ? "true" : "false");
            return this;
        }

        public JsonWriter Value(int v)
        {
            BeforeValue();
            _sb.Append(v.ToString(CultureInfo.InvariantCulture));
            return this;
        }

        public JsonWriter Value(long v)
        {
            BeforeValue();
            _sb.Append(v.ToString(CultureInfo.InvariantCulture));
            return this;
        }

        public JsonWriter Value(float v)
        {
            BeforeValue();
            if (float.IsNaN(v) || float.IsInfinity(v)) _sb.Append("null");
            else AppendNumber(v.ToString("R", CultureInfo.InvariantCulture));
            return this;
        }

        public JsonWriter Value(double v)
        {
            BeforeValue();
            if (double.IsNaN(v) || double.IsInfinity(v)) _sb.Append("null");
            else AppendNumber(v.ToString("R", CultureInfo.InvariantCulture));
            return this;
        }

        public JsonWriter Value(int? v) => v.HasValue ? Value(v.Value) : Null();

        public JsonWriter Value(float? v) => v.HasValue ? Value(v.Value) : Null();

        /// <summary>Writes strings, numbers, bools, null, string-keyed dictionaries and lists.</summary>
        public JsonWriter Any(object v)
        {
            switch (v)
            {
                case null: return Null();
                case string s: return Value(s);
                case bool b: return Value(b);
                case int i: return Value(i);
                case long l: return Value(l);
                case float f: return Value(f);
                case double d: return Value(d);
                case byte by: return Value((int)by);
                case IDictionary<string, object> dict:
                    BeginObject();
                    foreach (KeyValuePair<string, object> kv in dict)
                    {
                        Key(kv.Key);
                        Any(kv.Value);
                    }
                    return EndObject();
                case IEnumerable items:
                    BeginArray();
                    foreach (object item in items) Any(item);
                    return EndArray();
                default:
                    return Value(Convert.ToString(v, CultureInfo.InvariantCulture));
            }
        }

        // Keeps floats recognisable as floats ("8.0", not "8") for readers that check types.
        void AppendNumber(string s)
        {
            _sb.Append(s);
            if (s.IndexOfAny(FloatMarks) < 0) _sb.Append(".0");
        }

        void AppendString(string s)
        {
            _sb.Append('"');
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': _sb.Append("\\\""); break;
                    case '\\': _sb.Append("\\\\"); break;
                    case '\n': _sb.Append("\\n"); break;
                    case '\r': _sb.Append("\\r"); break;
                    case '\t': _sb.Append("\\t"); break;
                    default:
                        if (c < 0x20) _sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else _sb.Append(c);
                        break;
                }
            }
            _sb.Append('"');
        }

        void BeforeValue()
        {
            if (_afterKey)
            {
                _afterKey = false;
                return;
            }
            Separator();
        }

        void Separator()
        {
            if (_depth == 0) return;
            if (_hasItems[_depth - 1]) _sb.Append(", ");
            _hasItems[_depth - 1] = true;
        }

        void Push()
        {
            if (_depth == MaxDepth) throw new InvalidOperationException("JSON nested too deeply.");
            _hasItems[_depth++] = false;
        }

        void Pop()
        {
            if (_depth == 0) throw new InvalidOperationException("Unbalanced JSON end.");
            _depth--;
            _afterKey = false;
        }
    }

    /// <summary>
    /// Reads single values out of the flat JSON lines this package writes (steps.jsonl, meta.json).
    /// Not a general parser: it only looks up keys of the outermost object.
    /// </summary>
    public static class JsonScan
    {
        /// <summary>Index of the first character of the value of a top-level key, or -1.</summary>
        public static int FindTopLevelValue(string json, string key)
        {
            int depth = 0;
            for (int i = 0; i < json.Length; i++)
            {
                char c = json[i];
                if (c == '"')
                {
                    int end = SkipString(json, i);
                    if (depth == 1)
                    {
                        int j = SkipWhitespace(json, end + 1);
                        int start = i + 1;
                        if (j < json.Length && json[j] == ':' && end - start == key.Length &&
                            string.CompareOrdinal(json, start, key, 0, key.Length) == 0)
                            return SkipWhitespace(json, j + 1);
                    }
                    i = end;
                }
                else if (c == '{' || c == '[') depth++;
                else if (c == '}' || c == ']') depth--;
            }
            return -1;
        }

        public static bool TryGetLong(string json, string key, out long value)
        {
            value = 0;
            int i = FindTopLevelValue(json, key);
            return i >= 0 && long.TryParse(ReadToken(json, i), NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
        }

        public static bool TryGetFloatPair(string json, string key, out float a, out float b)
        {
            a = b = 0f;
            int i = FindTopLevelValue(json, key);
            if (i < 0 || json[i] != '[') return false;
            i = SkipWhitespace(json, i + 1);
            string first = ReadToken(json, i);
            i = SkipWhitespace(json, i + first.Length);
            if (i >= json.Length || json[i] != ',') return false;
            i = SkipWhitespace(json, i + 1);
            string second = ReadToken(json, i);
            return float.TryParse(first, NumberStyles.Float, CultureInfo.InvariantCulture, out a) &&
                   float.TryParse(second, NumberStyles.Float, CultureInfo.InvariantCulture, out b);
        }

        /// <summary>The raw JSON text of a top-level value, e.g. the whole "state" object.</summary>
        public static bool TryGetRaw(string json, string key, out string raw)
        {
            raw = null;
            int i = FindTopLevelValue(json, key);
            if (i < 0 || i >= json.Length) return false;
            char c = json[i];
            if (c == '"')
            {
                raw = json.Substring(i, SkipString(json, i) - i + 1);
                return true;
            }
            if (c != '{' && c != '[')
            {
                raw = ReadToken(json, i);
                return true;
            }
            int depth = 0;
            for (int j = i; j < json.Length; j++)
            {
                char d = json[j];
                if (d == '"') j = SkipString(json, j);
                else if (d == '{' || d == '[') depth++;
                else if ((d == '}' || d == ']') && --depth == 0)
                {
                    raw = json.Substring(i, j - i + 1);
                    return true;
                }
            }
            return false;
        }

        static int SkipString(string json, int openQuote)
        {
            for (int j = openQuote + 1; j < json.Length; j++)
            {
                if (json[j] == '\\') j++;
                else if (json[j] == '"') return j;
            }
            return json.Length - 1;
        }

        static int SkipWhitespace(string json, int i)
        {
            while (i < json.Length && char.IsWhiteSpace(json[i])) i++;
            return i;
        }

        static string ReadToken(string json, int i)
        {
            int end = i;
            while (end < json.Length && ",}] \t\r\n".IndexOf(json[end]) < 0) end++;
            return json.Substring(i, end - i);
        }
    }
}
