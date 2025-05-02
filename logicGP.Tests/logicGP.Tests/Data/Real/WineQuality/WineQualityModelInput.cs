using Microsoft.ML.Data;

namespace logicGP.Tests.Data.Real;

public class WineQualityModelInput
{
    [LoadColumn(0)]
    [ColumnName(@"fixed_acidity")]
    public float Fixed_acidity { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"volatile_acidity")]
    public float Volatile_acidity { get; set; }

    [LoadColumn(2)]
    [ColumnName(@"citric_acid")]
    public float Citric_acid { get; set; }

    [LoadColumn(3)]
    [ColumnName(@"residual_sugar")]
    public float Residual_sugar { get; set; }

    [LoadColumn(4)]
    [ColumnName(@"chlorides")]
    public float Chlorides { get; set; }

    [LoadColumn(5)]
    [ColumnName(@"free_sulfur_dioxide")]
    public float Free_sulfur_dioxide { get; set; }

    [LoadColumn(6)]
    [ColumnName(@"total_sulfur_dioxide")]
    public float Total_sulfur_dioxide { get; set; }

    [LoadColumn(7)]
    [ColumnName(@"density")]
    public float Density { get; set; }

    [LoadColumn(8)] [ColumnName(@"pH")] public float PH { get; set; }

    [LoadColumn(9)]
    [ColumnName(@"sulphates")]
    public float Sulphates { get; set; }

    [LoadColumn(10)]
    [ColumnName(@"alcohol")]
    public float Alcohol { get; set; }

    [LoadColumn(11)]
    [ColumnName(@"quality")]
    public uint Quality { get; set; }
}

public class WineQualityModelInputOriginal
{
    [LoadColumn(0)]
    [ColumnName(@"fixed_acidity")]
    public float Fixed_acidity { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"volatile_acidity")]
    public float Volatile_acidity { get; set; }

    [LoadColumn(2)]
    [ColumnName(@"citric_acid")]
    public float Citric_acid { get; set; }

    [LoadColumn(3)]
    [ColumnName(@"residual_sugar")]
    public float Residual_sugar { get; set; }

    [LoadColumn(4)]
    [ColumnName(@"chlorides")]
    public float Chlorides { get; set; }

    [LoadColumn(5)]
    [ColumnName(@"free_sulfur_dioxide")]
    public float Free_sulfur_dioxide { get; set; }

    [LoadColumn(6)]
    [ColumnName(@"total_sulfur_dioxide")]
    public float Total_sulfur_dioxide { get; set; }

    [LoadColumn(7)]
    [ColumnName(@"density")]
    public float Density { get; set; }

    [LoadColumn(8)] [ColumnName(@"pH")] public float PH { get; set; }

    [LoadColumn(9)]
    [ColumnName(@"sulphates")]
    public float Sulphates { get; set; }

    [LoadColumn(10)]
    [ColumnName(@"alcohol")]
    public float Alcohol { get; set; }

    [LoadColumn(11)]
    [ColumnName(@"quality")]
    public float Quality { get; set; }
}