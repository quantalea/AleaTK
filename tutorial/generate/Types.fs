module Types

open Fake
open Util

type LangType = 
| CSharp
| FSharp
| VB

type Tag = 
| Computations
| Regression 
| Classification 
| ConvolutionalNeuralNets
| RecurrentNeuralNets 
| ComputerVision 
| Image 
| Video 
| Speech 
| Text
| OptimizationTechniques

type MetaData =
    {
        Id : string
        Title : string
        Language : LangType
        Tags : list<Tag>
        SourceCodeLink : string
        GitLink : string
        ImageLink : string
        Src : string
    }

    // the folder name can be used as page name
    member this.Folder = filename this.Src

    // the extended doc link for a Src location `samples/csharp/CopyGeneric`  
    // is the path `samples/csharp/copygeneric.html`
    member this.ExtendedDocFile =
        let pagename = normalizeName this.Folder + ".html"
        (directory this.Src) @@ pagename

    member this.ExtendedDocLink =
        this.ExtendedDocFile.Replace(@"\", @"/")

    member this.Abstract = this.Src @@ "Readme.md"

    

