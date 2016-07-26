module SampleProjects 

open Fake
open Types

let languages =
    [
        CSharp,         "C#"
        FSharp,         "F#"
        VB,             "VB"    
    ]

let tags = 
    [
        Computations,               "General Purpose Computations" 
        Regression,                 "Regression" 
        Classification,             "Classification" 
        ConvolutionalNeuralNets,    "Convolutional Neural Nets" 
        RecurrentNeuralNets,        "Recurrent Neural Nets"
        ComputerVision,             "Computer Vision" 
        LanguageModelling,          "Natural Language Modelling" 
        Image,                      "Image" 
        Video,                      "Video"
        Speech,                     "Speech"
        Text,                       "Text"
        OptimizationTechniques,     "Optimization Techniques"
    ]

let gitLinkRoot = "http://github.com/quantalea/AleaTK/tree/master/tutorial/samples/"
let sourceCodeLinkCsharp name = "../samples/" + name + ".zip"

// Add all projects here so that html sample gallery builder can generate html code
let metaData = 
    [
        {   
            Id = "0001"
            Title = "Monte Carlo Pi Estimation"
            Language = CSharp
            Tags = [Computations]
            SourceCodeLink = sourceCodeLinkCsharp "MonteCarloPi"
            GitLink = gitLinkRoot + "MonteCarloPi"
            ImageLink = "images/montecarlo_pi.gif"
            Src = "samples" @@ "MonteCarloPi"
        }

        {   
            Id = "0002"
            Title = "MNIST Digits Classificiation"
            Language = CSharp
            Tags = [ComputerVision; Regression; Classification; ConvolutionalNeuralNets]
            SourceCodeLink = sourceCodeLinkCsharp "MNIST"
            GitLink = gitLinkRoot + "MNIST"
            ImageLink = "images/mnist.png"
            Src = "samples" @@ "MNIST"
        }

        {   
            Id = "0003"
            Title = "PTB Natural Language Modelling"
            Language = CSharp
            Tags = [ComputerVision; Regression; Classification; RecurrentNeuralNets]
            SourceCodeLink = sourceCodeLinkCsharp "PTB"
            GitLink = gitLinkRoot + "PTB"
            ImageLink = "images/mnist.png"
            Src = "samples" @@ "PTB"
        }
    ]