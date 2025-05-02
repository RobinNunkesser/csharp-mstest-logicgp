using System.Globalization;
using System.Text;
using Italbytz.Adapters.Algorithms.AI.Util;
using Italbytz.Ports.Algorithms.AI.Search.GP.SearchSpace;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;

/// <summary>
///     A class representing a monomial in the LogicGP algorithm.
///     It implements the IMonomial interface and provides methods for generating
///     predictions,
///     comparing monomials, and manipulating literals.
/// </summary>
/// <typeparam name="TCategory">The type of the categories used in the monomial.</typeparam>
/// <remarks>
///     The LogicGpMonomial class is used to represent a monomial in the LogicGP
///     algorithm.
///     It contains a list of literals that represent the monomial and provides
///     methods for generating predictions based on the literals.
///     The class supports different types of literals and provides methods for
///     manipulating literals, such as inserting, deleting, and replacing literals.
///     The class also provides methods for updating predictions and checking the
///     size of the monomial.
/// </remarks>
/// <seealso cref="IMonomial{TCategory}" />
public class LogicGpMonomial<TCategory> : IMonomial<TCategory>
{
    private readonly int _classes;
    private readonly LogicGpAlgorithm.Weighting? _usedWeighting;

    public LogicGpMonomial(IEnumerable<ILiteral<TCategory>> literals,
        int classes, List<string>? outputValues, List<string> labels,
        LogicGpAlgorithm.Weighting? usedWeighting)
    {
        ArgumentNullException.ThrowIfNull(outputValues);
        ArgumentNullException.ThrowIfNull(labels);
        Labels = labels;
        OutputColumn = outputValues;
        Literals = literals.ToList();
        _classes = classes;
        _usedWeighting = usedWeighting;
        Weights = new float[_classes];
        Weights[_classes - 1] = 1;
        CounterWeights = new float[_classes];
        UpdatePredictions();
    }

    public List<string> Labels { get; set; }

    public List<string>? OutputColumn { get; set; }

    public float[] CounterWeights { get; set; }

    public float[][] Predictions { get; set; }

    public int Size => Literals.Count;

    public void RandomizeWeights(bool restricted)
    {
        var random = ThreadSafeRandomNetCore.LocalRandom;
        if (restricted)
        {
            var index = random.Next(0, Weights.Length);
            for (var i = 0; i < Weights.Length; i++)
                Weights[i] = i == index ? 1 : 0;
        }
        else
        {
            for (var i = 0; i < Weights.Length; i++)
                Weights[i] = (float)random.NextDouble();
        }

        UpdatePredictions();
    }

    public object Clone()
    {
        return new LogicGpMonomial<TCategory>(Literals, _classes, OutputColumn,
            Labels, _usedWeighting)
        {
            Weights = new float[_classes].Select((_, i) => Weights[i]).ToArray()
        };
    }

    public List<ILiteral<TCategory>> Literals { get; set; }
    public float[] Weights { get; set; }

    public void UpdatePredictions()
    {
        var literalPredictions = Literals[0].Predictions;
        if (Literals.Count > 1)
            literalPredictions = Literals.Aggregate(literalPredictions,
                (current, literal) =>
                    current.Zip(literal.Predictions, (a, b) => a && b)
                        .ToArray());

        Predictions = new float[Literals[0].Predictions.Length][];

        // TODO: This is a hack for quick and dirty weight computation
        // It relies on the fact that training data has more rows than validation data
        if (Predictions.Length == OutputColumn.Count)
            switch (_usedWeighting)
            {
                case LogicGpAlgorithm.Weighting.Fixed:
                    Weights = [0.0f, 1.0f];
                    break;
                case LogicGpAlgorithm.Weighting.Computed:
                    ComputeWeights(literalPredictions);
                    break;
                case LogicGpAlgorithm.Weighting.Mutated:
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

        for (var i = 0; i < Predictions.Length; i++)
            if (literalPredictions[i])
                Predictions[i] = Weights;
            else
                Predictions[i] = CounterWeights;
    }

    private void ComputeWeights(bool[] literalPredictions)
    {
        var count = new int[_classes];
        var counterCount = new int[_classes];
        for (var i = 0; i < Predictions.Length; i++)
        {
            var index = Labels.IndexOf(OutputColumn[i]);
            if (index == -1 || index >= _classes)
                throw new InvalidOperationException();
            if (literalPredictions[i])
                count[index]++;
            else
                counterCount[index]++;
        }

        var inDistribution = count.Select(c => (float)c).ToArray();
        var outDistribution = counterCount.Select(c => (float)c).ToArray();
        var sum = inDistribution.Sum();
        if (sum == 0)
            sum = 1;
        for (var j = 0; j < inDistribution.Length; j++)
            inDistribution[j] /= sum;

        sum = outDistribution.Sum();
        if (sum == 0)
            sum = 1;
        for (var j = 0; j < outDistribution.Length; j++)
            outDistribution[j] /= sum;

        var newWeights = new float[_classes];
        for (var j = 0; j < newWeights.Length; j++)
            newWeights[j] = inDistribution[j] / outDistribution[j];

        var newCounterWeights = new float[_classes];
        for (var j = 0; j < newCounterWeights.Length; j++)
            newCounterWeights[j] = 0;

        Weights = newWeights;
        //CounterWeights = newCounterWeights;
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append("  ");
        sb.Append(string.Join(" |  ", Weights.Select(w => w.ToString("F2",
            CultureInfo.InvariantCulture))));
        sb.Append(" | ");
        sb.Append(string.Join("", Literals));
        sb.Append(" |");
        return sb.ToString();
    }

    public float[] Predict<TSrc>(TSrc src) where TSrc : class, new()
    {
        var literalPredictions =
            ((LogicGpLiteral<string>)Literals[0]).Predict(src);
        if (Literals.Count > 1)
            literalPredictions = Literals.Aggregate(literalPredictions,
                (current, literal) =>
                    current && ((LogicGpLiteral<string>)literal).Predict(src)
            );

        return literalPredictions ? Weights : new float[_classes];
    }
}