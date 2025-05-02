using System.Globalization;
using System.Text;
using Italbytz.ML;
using Italbytz.Ports.Algorithms.AI.Search.GP.SearchSpace;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;

/// <summary>
///     A class representing a literal in the LogicGP algorithm.
///     It implements the ILiteral interface and provides methods for generating
///     predictions,
///     comparing literals, and converting them to string representations.
/// </summary>
/// <typeparam name="TCategory">The type of the categories used in the literal.</typeparam>
/// <remarks>
///     The LogicGpLiteral class is used to represent a literal in the LogicGP
///     algorithm.
///     It contains a bit set that represents the categories associated with the
///     literal.
///     The class provides methods for generating predictions based on the bit set,
///     comparing literals, and converting them to string representations.
///     The class supports different types of literals, including Dussault, Rudell,
///     Su, and LessGreater.
/// </remarks>
/// <seealso cref="ILiteral{TCategory}" />
public class LogicGpLiteral<TCategory> : ILiteral<TCategory>
    where TCategory : class
{
    private readonly bool[] _bitSet;
    private readonly int _featureColumn;
    private readonly List<TCategory> _orderedCategories;

    public LogicGpLiteral(int featureColumn, string label,
        HashSet<TCategory> categories, int set,
        List<TCategory> trainingData,
        LogicGpLiteralType literalType = LogicGpLiteralType.Rudell)
    {
        _featureColumn = featureColumn;
        Label = label;
        Set = set;
        LiteralType = literalType;
        _orderedCategories = categories.OrderBy(c => c).ToList();
        _bitSet = new bool[_orderedCategories.Count];
        for (var i = 0; i < _orderedCategories.Count; i++)
            _bitSet[i] = (set & (1 << i)) != 0;
        GeneratePredictions(trainingData);
    }

    private int Set { get; }
    private LogicGpLiteralType LiteralType { get; }

    public string Label { get; set; }

    public bool[] Predictions { get; set; }

    public void GeneratePredictions(List<TCategory> data)
    {
        Predictions = new bool[data.Count];
        for (var i = 0; i < data.Count; i++)
        {
            var category = data[i];
            var index = _orderedCategories.IndexOf(category);
            Predictions[i] =
                index > -1 && index < _bitSet.Length && _bitSet[index];
        }
    }

    public int CompareTo(ILiteral<TCategory>? other)
    {
        return Compare(this, other);
    }

    private static int Compare(ILiteral<TCategory>? x, ILiteral<TCategory>? y)
    {
        if (x is null && y is null) return 0;
        if (x is not LogicGpLiteral<TCategory> literal1) return -1;
        if (y is not LogicGpLiteral<TCategory> literal2) return 1;
        if (x.Label != y.Label)
            return string.Compare(x.Label, y.Label, StringComparison.Ordinal);
        if (literal1.Set !=
            literal2.Set)
            return literal1.Set.CompareTo(
                literal2.Set);
        return 0;
    }


    public override bool Equals(object? obj)
    {
        if (obj is null) return false;
        if (ReferenceEquals(this, obj)) return true;
        if (obj.GetType() != GetType()) return false;
        if (obj is not LogicGpLiteral<TCategory> other) return false;
        if (other.LiteralType != LiteralType) return false;
        if (other.Label != Label) return false;
        return other.Set == Set;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(_bitSet, _orderedCategories, Label);
    }

    public override string ToString()
    {
        switch (LiteralType)
        {
            case LogicGpLiteralType.Dussault:
                return ToDussaultString();
            case LogicGpLiteralType.Rudell:
                return ToRudellString();
            case LogicGpLiteralType.Su:
                return ToSuString();
            case LogicGpLiteralType.LessGreater:
                return ToLessGreaterString();
            default:
                throw new ArgumentOutOfRangeException();
        }
    }

    private string ToLessGreaterString()
    {
        var sb = new StringBuilder();
        if (_bitSet[0])
        {
            var index = Array.IndexOf(_bitSet, false);
            sb.Append($"({Label} < {_orderedCategories[index]})");
        }
        else
        {
            var index = Array.IndexOf(_bitSet, true);
            sb.Append($"({Label} > {_orderedCategories[index - 1]})");
        }

        return sb.ToString();
    }

    private string ToSuString()
    {
        var sb = new StringBuilder();
        var firstIndexPositive = Array.IndexOf(_bitSet, true);
        if (firstIndexPositive == -1)
            throw new ArgumentException("No positive value in BitSet");
        var firstIndexNegative = Array.IndexOf(_bitSet, false);
        if (firstIndexNegative == -1)
            throw new ArgumentException("No negative value in BitSet");
        var lastIndexPositive = Array.LastIndexOf(_bitSet, true);
        var lastIndexNegative = Array.LastIndexOf(_bitSet, false);
        var negative = false;
        for (var i = firstIndexPositive; i < lastIndexPositive; i++)
            if (!_bitSet[i])
                negative = true;
        if (negative)
            sb.Append(
                $"({Label} ∉ [{_orderedCategories[firstIndexNegative]},{_orderedCategories[lastIndexNegative]}])");
        else
            sb.Append(
                $"({Label} ∈ [{_orderedCategories[firstIndexPositive]},{_orderedCategories[lastIndexPositive]}])");
        return sb.ToString();
    }

    private string ToDussaultString()
    {
        var sb = new StringBuilder();
        var count = _bitSet.Count(bit => bit);
        if (count != 1 && count != _bitSet.Length - 1)
            throw new ArgumentException(
                "Dussault literals must have exactly one or all but one bit set");
        if (count == 1)
            sb.Append(
                $"({Label} = {_orderedCategories[Array.IndexOf(_bitSet, true)]})");
        else
            sb.Append(
                $"({Label} \u2260 {_orderedCategories[Array.IndexOf(_bitSet, false)]})");
        return sb.ToString();
    }

    private string ToRudellString()
    {
        var sb = new StringBuilder();
        sb.Append($"({Label} ∈ {{");
        for (var j = 0; j < _orderedCategories.Count; j++)
            if (_bitSet[j])
                sb.Append(_orderedCategories[j] + ",");
        sb.Remove(sb.Length - 1, 1);
        sb.Append("})");
        return sb.ToString();
    }

    public bool Predict<TSrc>(TSrc src) where TSrc : class, new()
    {
        var rawCategory = src switch
        {
            BinaryClassificationInputSchema binaryFeatureRow => binaryFeatureRow
                .Features[_featureColumn],
            MulticlassClassificationInputSchema multiclassFeatureRow =>
                multiclassFeatureRow.Features[_featureColumn],
            _ => throw new InvalidDataException()
        };

        var category =
            rawCategory.ToString(CultureInfo.InvariantCulture) as TCategory;
        var index = _orderedCategories.IndexOf(category);
        return index > -1 && index < _bitSet.Length && _bitSet[index];
    }
}