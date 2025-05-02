using Italbytz.Adapters.Algorithms.AI.Search.GP.Control;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Fitness;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Initialization;
using Italbytz.Adapters.Algorithms.AI.Search.GP.PopulationManager;
using Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Selection;
using Italbytz.Adapters.Algorithms.AI.Search.GP.StoppingCriterion;
using Italbytz.Ports.Algorithms.AI.Search.GP;
using Italbytz.Ports.Algorithms.AI.Search.GP.Fitness;
using Microsoft.Extensions.DependencyInjection;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP;

public static class Dependencies
{
    public static IServiceCollection AddServices(
        this IServiceCollection services)
    {
        services
            .AddScoped<IGeneticProgram,
                GeneticProgram>();
        services.AddScoped<LogicGpAlgorithm>();
        services.AddScoped<LogicGpGpasBinaryTrainer>();
        services.AddScoped<LogicGpFlrwMacroMulticlassTrainer>();
        services.AddScoped<LogicGpFlrwMicroMulticlassTrainer>();
        services.AddScoped<RandomInitialization>();
        services.AddScoped<CompleteInitialization>();
        services.AddScoped<GenerationStoppingCriterion>();
        services.AddScoped<UniformSelection>();
        services.AddScoped<ParetoFrontSelection>();
        services.AddScoped<IFitnessFunction, LogicGpPareto>();
        services.AddScoped<DefaultPopulationManager>();
        services.AddScoped<LogicGpSearchSpace>();
        services.AddScoped<DataManager>();
        return services;
    }
}