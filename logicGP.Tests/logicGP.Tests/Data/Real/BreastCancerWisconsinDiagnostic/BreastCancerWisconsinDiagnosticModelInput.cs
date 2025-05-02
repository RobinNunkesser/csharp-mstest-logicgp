using Microsoft.ML.Data;

namespace logicGP.Tests.Data.Real;

public class BreastCancerWisconsinDiagnosticModelInput
{
    [LoadColumn(0)]
    [ColumnName(@"radius1")]
    public float Radius1 { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"texture1")]
    public float Texture1 { get; set; }

    [LoadColumn(2)]
    [ColumnName(@"perimeter1")]
    public float Perimeter1 { get; set; }

    [LoadColumn(3)] [ColumnName(@"area1")] public float Area1 { get; set; }

    [LoadColumn(4)]
    [ColumnName(@"smoothness1")]
    public float Smoothness1 { get; set; }

    [LoadColumn(5)]
    [ColumnName(@"compactness1")]
    public float Compactness1 { get; set; }

    [LoadColumn(6)]
    [ColumnName(@"concavity1")]
    public float Concavity1 { get; set; }

    [LoadColumn(7)]
    [ColumnName(@"concave_points1")]
    public float Concave_points1 { get; set; }

    [LoadColumn(8)]
    [ColumnName(@"symmetry1")]
    public float Symmetry1 { get; set; }

    [LoadColumn(9)]
    [ColumnName(@"fractal_dimension1")]
    public float Fractal_dimension1 { get; set; }

    [LoadColumn(10)]
    [ColumnName(@"radius2")]
    public float Radius2 { get; set; }

    [LoadColumn(11)]
    [ColumnName(@"texture2")]
    public float Texture2 { get; set; }

    [LoadColumn(12)]
    [ColumnName(@"perimeter2")]
    public float Perimeter2 { get; set; }

    [LoadColumn(13)]
    [ColumnName(@"area2")]
    public float Area2 { get; set; }

    [LoadColumn(14)]
    [ColumnName(@"smoothness2")]
    public float Smoothness2 { get; set; }

    [LoadColumn(15)]
    [ColumnName(@"compactness2")]
    public float Compactness2 { get; set; }

    [LoadColumn(16)]
    [ColumnName(@"concavity2")]
    public float Concavity2 { get; set; }

    [LoadColumn(17)]
    [ColumnName(@"concave_points2")]
    public float Concave_points2 { get; set; }

    [LoadColumn(18)]
    [ColumnName(@"symmetry2")]
    public float Symmetry2 { get; set; }

    [LoadColumn(19)]
    [ColumnName(@"fractal_dimension2")]
    public float Fractal_dimension2 { get; set; }

    [LoadColumn(20)]
    [ColumnName(@"radius3")]
    public float Radius3 { get; set; }

    [LoadColumn(21)]
    [ColumnName(@"texture3")]
    public float Texture3 { get; set; }

    [LoadColumn(22)]
    [ColumnName(@"perimeter3")]
    public float Perimeter3 { get; set; }

    [LoadColumn(23)]
    [ColumnName(@"area3")]
    public float Area3 { get; set; }

    [LoadColumn(24)]
    [ColumnName(@"smoothness3")]
    public float Smoothness3 { get; set; }

    [LoadColumn(25)]
    [ColumnName(@"compactness3")]
    public float Compactness3 { get; set; }

    [LoadColumn(26)]
    [ColumnName(@"concavity3")]
    public float Concavity3 { get; set; }

    [LoadColumn(27)]
    [ColumnName(@"concave_points3")]
    public float Concave_points3 { get; set; }

    [LoadColumn(28)]
    [ColumnName(@"symmetry3")]
    public float Symmetry3 { get; set; }

    [LoadColumn(29)]
    [ColumnName(@"fractal_dimension3")]
    public float Fractal_dimension3 { get; set; }

    [LoadColumn(30)]
    [ColumnName(@"Diagnosis")]
    public string Diagnosis { get; set; }
}