using Italbytz.Adapters.Algorithms.AI.Search.GP.Control;
using Italbytz.Adapters.Algorithms.AI.Util;
using Italbytz.Ports.Algorithms.AI.Search.GP.Individuals;
using Italbytz.Ports.Algorithms.AI.Search.GP.SearchSpace;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;

/// <summary>
///     A class representing a genotype in the LogicGP algorithm.
///     It implements the IGenotype interface and provides methods for generating
///     predictions,
///     comparing genotypes, and manipulating monomials.
/// </summary>
/// <remarks>
///     The LogicGpGenotype class is used to represent a genotype in the LogicGP
///     algorithm.
///     It contains a polynomial that represents the genotype and provides methods
///     for generating predictions based on the polynomial.
///     The class supports different types of monomials and provides methods for
///     manipulating monomials, such as inserting, deleting, and replacing
///     literals.
///     The class also provides methods for updating predictions and checking if
///     the
///     genotype is empty.
/// </remarks>
/// <seealso cref="IGenotype" />
public class LogicGpGenotype : IGenotype
{
    private readonly DataManager _data;
    private readonly IPolynomial<string> _polynomial;
    private readonly LogicGpAlgorithm.Weighting _usedWeighting;
    private string[]? _predictedClasses;

    public LogicGpGenotype(int classes, DataManager data,
        List<string>? outputColumn, List<string> labels,
        LogicGpAlgorithm.Weighting? usedWeighting)
    {
        ArgumentNullException.ThrowIfNull(outputColumn);
        ArgumentNullException.ThrowIfNull(labels);
        Labels = labels;
        OutputColumn = outputColumn;
        _data = data;
        _usedWeighting = usedWeighting ?? LogicGpAlgorithm.Weighting.Computed;
        var literal = data.GetRandomLiteral();
        var monomial =
            new LogicGpMonomial<string>([literal], classes, OutputColumn,
                Labels, _usedWeighting);
        _polynomial = new LogicGpPolynomial<string>([monomial], classes,
            OutputColumn,
            Labels, _usedWeighting);
    }

    public LogicGpGenotype(IPolynomial<string> polynomial,
        DataManager data,
        List<string>? outputColumn, List<string> labels,
        LogicGpAlgorithm.Weighting? usedWeighting)
    {
        ArgumentNullException.ThrowIfNull(outputColumn);
        ArgumentNullException.ThrowIfNull(labels);
        _data = data;
        _polynomial = polynomial;
        OutputColumn = outputColumn;
        Labels = labels;
        _usedWeighting = usedWeighting ?? LogicGpAlgorithm.Weighting.Computed;
    }

    public List<string> Labels { get; set; }

    public List<string>? OutputColumn { get; set; }

    public string[] PredictedClasses
    {
        get
        {
            if (_predictedClasses == null) UpdatePredictedClasses();
            return _predictedClasses;
        }
        set => _predictedClasses = value;
    }


    public float[][] Predictions => _polynomial.Predictions;

    public double[]? LatestKnownFitness { get; set; }

    public int Size => _polynomial.Size;

    public void UpdatePredictions()
    {
        LatestKnownFitness = null;
        _polynomial.UpdatePredictions();
        UpdatePredictedClasses();
    }

    public object Clone()
    {
        return new LogicGpGenotype(
            (IPolynomial<string>)_polynomial.Clone(),
            _data, OutputColumn, Labels, _usedWeighting);
    }

    private void UpdatePredictedClasses()
    {
        var predictionStrategy =
            LogicGpAlgorithm.PredictionStrategy.Max;
        _predictedClasses = new string[Predictions.Length];
        for (var i = 0; i < Predictions.Length; i++)
        {
            var index = predictionStrategy switch
            {
                LogicGpAlgorithm.PredictionStrategy.Max => MaxIndex(i),
                LogicGpAlgorithm.PredictionStrategy.SoftmaxProbability =>
                    SoftmaxProbabilityIndex(i),
                _ => throw new ArgumentOutOfRangeException()
            };

            _predictedClasses[i] = _data.Labels[index];
        }
    }

    private int SoftmaxProbabilityIndex(int i)
    {
        var random = ThreadSafeRandomNetCore.LocalRandom;
        var cumulative = Predictions[i]
            .Select((value, index) => new { value, index })
            .Select((x, index) => new
            {
                x.index, cumulative = Predictions[i].Take(index + 1).Sum()
            })
            .ToList();
        var randomValue = random.NextDouble();
        var chosen =
            cumulative.FirstOrDefault(x => x.cumulative >= randomValue);
        return chosen?.index ?? 0;
    }

    private int MaxIndex(int i)
    {
        return Array.IndexOf(Predictions[i], Predictions[i].Max());
    }


    public override string ToString()
    {
        return _polynomial.ToString() ?? string.Empty;
    }

    public IMonomial<string> GetRandomMonomial()
    {
        return _polynomial.GetRandomMonomial();
    }

    public void InsertMonomial(IMonomial<string> monomial)
    {
        _polynomial.Monomials.Add(monomial);
        UpdatePredictions();
    }

    public void RandomizeAMonomialWeight(bool restricted)
    {
        var monomial = GetRandomMonomial();
        monomial.RandomizeWeights(restricted);
        UpdatePredictions();
    }

    public void DeleteRandomLiteral()
    {
        var monomial = GetRandomMonomial();
        monomial.Literals.RemoveAt(
            ThreadSafeRandomNetCore.LocalRandom.Next(monomial.Literals.Count));
        if (monomial.Literals.Count == 0)
            _polynomial.Monomials.Remove(monomial);
        else
            monomial.UpdatePredictions();
        UpdatePredictions();
    }

    public bool IsEmpty()
    {
        return _polynomial.Monomials.Count == 0;
    }

    public void DeleteRandomMonomial()
    {
        _polynomial.Monomials.RemoveAt(
            ThreadSafeRandomNetCore.LocalRandom.Next(
                _polynomial.Monomials.Count));
        if (_polynomial.Monomials.Count > 0)
            UpdatePredictions();
    }

    public void InsertRandomLiteral()
    {
        var monomial = GetRandomMonomial();
        monomial.Literals.Add(_data.GetRandomLiteral());
        monomial.UpdatePredictions();
        UpdatePredictions();
    }

    public void InsertRandomMonomial()
    {
        _polynomial.Monomials.Add(new LogicGpMonomial<string>(
            new List<ILiteral<string>>
                { _data.GetRandomLiteral() },
            _polynomial.Monomials[0].Weights.Length, OutputColumn, Labels,
            _usedWeighting));
        UpdatePredictions();
    }

    public void ReplaceRandomLiteral()
    {
        var monomial = GetRandomMonomial();
        monomial.Literals[
                ThreadSafeRandomNetCore.LocalRandom.Next(
                    monomial.Literals.Count)] =
            _data.GetRandomLiteral();
        monomial.UpdatePredictions();
        UpdatePredictions();
    }

    public void UpdatePredictionsRecursively()
    {
        foreach (var monomial in _polynomial.Monomials)
            monomial.UpdatePredictions();
        _polynomial.UpdatePredictions();
        UpdatePredictedClasses();
    }

    public string LiteralSignature()
    {
        var literals = _polynomial.GetAllLiterals();
        literals.Sort();
        return string.Join(" ", literals.Select(literal => literal.Label));
    }

    public bool IsLiterallyEqual(LogicGpGenotype other)
    {
        var literals = _polynomial.GetAllLiterals();
        var otherLiterals = other._polynomial.GetAllLiterals();
        if (literals.Count != otherLiterals.Count) return false;

        return !literals.Except(otherLiterals).Any() &&
               !otherLiterals.Except(literals).Any();
    }

    public TDst Predict<TSrc, TDst>(TSrc src) where TDst : class, new()
        where TSrc : class, new()
    {
        return ((LogicGpPolynomial<string>)_polynomial)
            .Predict<TSrc, TDst>(src);
    }
}