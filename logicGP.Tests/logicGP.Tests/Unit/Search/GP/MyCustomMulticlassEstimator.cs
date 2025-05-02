using Italbytz.ML;
using Italbytz.Adapters.Algorithms.AI.Util.ML;
using Microsoft.ML;

namespace logicGP.Tests.Unit.Search.GP;

public class MyCustomMulticlassEstimator : IEstimator<ITransformer>
{
    public ITransformer Fit(IDataView input)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        return mlContext.Transforms.CustomMapping(
            MyCustomMulticlassMapper
                .GetMapping<BinaryClassificationInputSchema,
                    TernaryClassificationClassificationOutputSchema>(),
            null).Fit(input);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;

        /* var outputSchema = mlContext.MulticlassClassification.Trainers
             .SdcaMaximumEntropy().GetOutputSchema(inputSchema);
         return outputSchema;*/

        var outputSchema = mlContext.Transforms.CustomMapping(
            MyCustomMulticlassMapper
                .GetMapping<MulticlassClassificationInputSchema,
                    TernaryClassificationClassificationOutputSchema>(),
            null).GetOutputSchema(inputSchema);

        return outputSchema;

        /*var outColumns = outputSchema.ToDictionary(x => x.Name);
        var scoreColumn = outColumns["Score"];


        return new SchemaShape(outColumns.Values);*/


        /*
        var outColumns = inputSchema.ToDictionary(x => x.Name);
        return new SchemaShape(outColumns.Values);
        */
    }

    /*
    private IEnumerable<SchemaShape.Column> GetOutputColumnsCore(SchemaShape inputSchema)
    {
        return new[]
        {
            new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(GetTrainerOutputAnnotation())),
            new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(GetTrainerOutputAnnotation(true))),
            new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(GetTrainerOutputAnnotation()))
        };
    }

    public IEnumerable<SchemaShape.Column> GetTrainerOutputAnnotation(bool isNormalized = false)
    {
        var cols = new List<SchemaShape.Column>();
        cols.Add(new SchemaShape.Column(Kinds.ScoreColumnSetId, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true));
        cols.Add(new SchemaShape.Column(Kinds.ScoreColumnKind, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false));
        cols.Add(new SchemaShape.Column(Kinds.ScoreValueKind, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false));
        if (isNormalized)
            cols.Add(new SchemaShape.Column(Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false));
        return cols;
    }
    */
}