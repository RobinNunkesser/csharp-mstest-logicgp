using Microsoft.ML.Data;

namespace logicGP.Tests.Data.Real;

public class IrisModelInput
{
    [LoadColumn(0)]
    [ColumnName(@"sepal length")]
    public float Sepal_length { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"sepal width")]
    public float Sepal_width { get; set; }

    [LoadColumn(2)]
    [ColumnName(@"petal length")]
    public float Petal_length { get; set; }

    [LoadColumn(3)]
    [ColumnName(@"petal width")]
    public float Petal_width { get; set; }

    [LoadColumn(4)] [ColumnName(@"class")] public string Class { get; set; }
}