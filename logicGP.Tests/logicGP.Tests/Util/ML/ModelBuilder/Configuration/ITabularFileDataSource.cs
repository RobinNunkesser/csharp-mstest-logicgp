namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

public interface ITabularFileDataSource : ITabularDataSource
{
    string? FilePath { get; set; }

    string? Delimiter { get; set; }

    char DecimalMarker { get; set; }

    bool HasHeader { get; set; }

    bool AllowQuoting { get; set; }

    char EscapeCharacter { get; set; }

    bool ReadMultiLines { get; set; }

    bool KeepDiacritics { get; set; }

    bool KeepPunctuations { get; set; }
}