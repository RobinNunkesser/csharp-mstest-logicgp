using System.Globalization;
using System.Text;
using Italbytz.Adapters.Algorithms.AI.Util;
using Italbytz.ML;
using Italbytz.Ports.Algorithms.AI.Learning.ML;
using Italbytz.Ports.Algorithms.AI.Search.GP.SearchSpace;
using Microsoft.ML.Data;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;

/// <summary>
///     A class representing a polynomial in the LogicGP algorithm.
///     It implements the IPolynomial interface and provides methods for generating
///     predictions and comparing polynomials.
/// </summary>
/// <typeparam name="TCategory">The type of the categories used in the polynomial.</typeparam>
/// <remarks>
///     The LogicGpPolynomial class is used to represent a polynomial in the
///     LogicGP
///     algorithm.
///     It contains a list of monomials that represent the polynomial and provides
///     methods for generating predictions based on the monomials.
///     The class also provides methods for updating predictions and checking the
///     size
///     of the polynomial.
/// </remarks>
/// <seealso cref="IPolynomial{TCategory}" />
public class LogicGpPolynomial<TCategory> : IPolynomial<TCategory>
{
    private readonly int _classes;
    private readonly List<string> _labels;
    private readonly List<string>? _outputValues;
    private readonly LogicGpAlgorithm.Weighting? _usedWeighting;

    public LogicGpPolynomial(IEnumerable<IMonomial<TCategory>> monomials,
        int classes, List<string>? outputValues, List<string> labels,
        LogicGpAlgorithm.Weighting? usedWeighting)
    {
        _classes = classes;
        _outputValues = outputValues;
        _labels = labels;
        Monomials = [..monomials];
        Weights = new float[Monomials[0].Literals[0].Predictions.Length];
        _usedWeighting = usedWeighting;
        ComputeWeights();
        UpdatePredictions();
    }

    public float[] Weights { get; set; }

    public float[][] Predictions { get; set; }

    public int Size
    {
        get { return Monomials.Sum(monomial => monomial.Size); }
    }

    public IMonomial<TCategory> GetRandomMonomial()
    {
        var random = ThreadSafeRandomNetCore.LocalRandom;
        return Monomials[random.Next(Monomials.Count)];
    }

    public object Clone()
    {
        var monomials =
            Monomials.Select(
                monomial => (IMonomial<TCategory>)monomial.Clone());
        return new LogicGpPolynomial<TCategory>(
            monomials, _classes, _outputValues, _labels, _usedWeighting);
    }

    public List<IMonomial<TCategory>> Monomials { get; set; }

    public void UpdatePredictions()
    {
        if (Monomials.Count == 0) return;
        Predictions = new float[Monomials[0].Literals[0].Predictions.Length][];
        var count = new int[_classes];
        for (var i = 0; i < Predictions.Length; i++)
        {
            // Aggregate the predictions from all monomials
            Predictions[i] = Monomials
                .Select(monomial => monomial.Predictions[i].ToArray())
                .Aggregate((a, b) =>
                    a.Zip(b, (c, d) => c + d).ToArray());

            // Count the number of data rows not covered by any monomial^
            if (Predictions.Length == _outputValues.Count &&
                Predictions[i].Sum() == 0)
                count[_labels.IndexOf(_outputValues[i])]++;

            // Normalize the predictions
            var sum = 0.0f;
            for (var j = 0; j < Predictions[i].Length; j++)
                sum += Predictions[i][j];

            for (var j = 0; j < Predictions[i].Length; j++)
                Predictions[i][j] /= sum;
        }

        // If there are any data rows not covered by any monomial,
        // compute a weight representing the distribution of classes
        // in the uncovered rows in relation to the covered rows
        if (count.Sum() > 0)
            ComputeWeightsForCount(count);

        // Set the weight w_0 for every row the monomials do not cover
        foreach (var pred in Predictions)
        {
            if (pred.Sum() != 0) continue;
            for (var j = 0; j < pred.Length; j++)
                pred[j] = Weights[j];
        }
    }

    public List<ILiteral<TCategory>> GetAllLiterals()
    {
        return Monomials.SelectMany(monomial => monomial.Literals).ToList();
    }

    private void ComputeWeightsForCount(int[] count)
    {
        if (_usedWeighting == LogicGpAlgorithm.Weighting.Fixed)
        {
            Weights = [1.0f, 0.0f];
            return;
        }

        var weights = count.Select(c => (float)c).ToArray();

        var sum = weights.Sum();
        if (sum == 0)
            sum = 1;
        for (var j = 0; j < weights.Length; j++)
            weights[j] /= sum;

        Weights = weights;
    }

    private void ComputeWeights()
    {
        var count = new int[_classes];
        foreach (var index in _outputValues.Select(value =>
                     _labels.IndexOf(value))) count[index]++;
        ComputeWeightsForCount(count);
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append("\n|");
        for (var i = 0; i < Weights.Length; i++) sb.Append($" $w_{i}$ |");
        sb.Append(" Condition                                   |\n|");
        for (var i = 0; i < Weights.Length; i++) sb.Append(" ----- |");
        sb.Append(" ------------------------------------------- |\n|  ");
        sb.Append(string.Join(" |  ", Weights.Select(w => w.ToString("F2",
            CultureInfo.InvariantCulture))));
        sb.Append(" | None below fulfilled                        |\n|");
        sb.Append(string.Join("\n|", Monomials));
        sb.Append('\n');
        return sb.ToString();
    }

    public TDst Predict<TSrc, TDst>(TSrc src) where TSrc : class, new()
        where TDst : class, new()
    {
        var scores = Monomials
            .Select(monomial =>
                ((LogicGpMonomial<string>)monomial).Predict(src))
            .Aggregate((a, b) =>
                a.Zip(b, (c, d) => c + d).ToArray());
        if (scores.Sum() == 0) scores = Weights;

        if (_classes == 2)
        {
            var prediction = new BinaryClassificationOutputSchema
            {
                Score = scores[1]
            };
            var probabilities = new float[scores.Length];
            var sum = scores.Sum();
            for (var j = 0; j < scores.Length; j++)
                probabilities[j] = scores[j] / sum;

            prediction.Probability = probabilities[1];
            // ToDo: Adapt to Mapping
            prediction.PredictedLabel = 1;
            return prediction as TDst ?? throw new InvalidOperationException();
        }
        else
        {
            ICustomMappingMulticlassClassificationOutputSchema prediction =
                _classes switch
                {
                    3 => new TernaryClassificationClassificationOutputSchema
                    {
                        Score = new VBuffer<float>(scores.Length, scores)
                    },
                    4 => new QuaternaryClassificationClassificationOutputSchema
                    {
                        Score = new VBuffer<float>(scores.Length, scores)
                    },
                    5 => new QuinaryClassificationClassificationOutputSchema
                    {
                        Score = new VBuffer<float>(scores.Length, scores)
                    },
                    6 => new SenaryClassificationClassificationOutputSchema
                    {
                        Score = new VBuffer<float>(scores.Length, scores)
                    },
                    7 => new SeptenaryClassificationClassificationOutputSchema
                    {
                        Score = new VBuffer<float>(scores.Length, scores)
                    },
                    8 => new OctonaryClassificationClassificationOutputSchema
                    {
                        Score = new VBuffer<float>(scores.Length, scores)
                    },
                    9 => new NonaryClassificationClassificationOutputSchema
                    {
                        Score = new VBuffer<float>(scores.Length, scores)
                    },
                    10 => new DenaryClassificationClassificationOutputSchema
                    {
                        Score = new VBuffer<float>(scores.Length, scores)
                    },
                    11 => new UndenaryClassificationClassificationOutputSchema
                    {
                        Score = new VBuffer<float>(scores.Length, scores)
                    },
                    12 => new DuodenaryClassificationClassificationOutputSchema
                    {
                        Score = new VBuffer<float>(scores.Length, scores)
                    },
                    13 => new TridenaryClassificationClassificationOutputSchema
                    {
                        Score = new VBuffer<float>(scores.Length, scores)
                    },
                    14 => new
                        TetradenaryClassificationClassificationOutputSchema
                        {
                            Score = new VBuffer<float>(scores.Length, scores)
                        },
                    15 => new
                        PentadenaryClassificationClassificationOutputSchema
                        {
                            Score = new VBuffer<float>(scores.Length, scores)
                        },

                    _ => throw new ArgumentOutOfRangeException(
                        $"The number of classes {_classes} is not supported.")
                };
            var probabilities = new float[scores.Length];
            var sum = scores.Sum();
            for (var j = 0; j < scores.Length; j++)
                probabilities[j] = scores[j] / sum;

            prediction.Probability =
                new VBuffer<float>(scores.Length, probabilities);


            prediction.PredictedLabel =
                (uint)Array.IndexOf(probabilities, probabilities.Max()) + 1;
            ;
            return prediction as TDst ?? throw new InvalidOperationException();
        }

        return null;
    }
}