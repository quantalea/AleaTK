## How To

### Build from Source
***

Check out the source from the repository. Alea TK uses [paket](https://fsprojects.github.io/Paket/index.html) to manage its project dependencies. After checkout run

```
.paket\paket.exe restore
```

to download the required NuGet packages. 

Install [CUDA 7.5](https://developer.nvidia.com/cuda-75-downloads-archive) and [cuDNN 5.1](https://developer.nvidia.com/rdp/cudnn-download) for CUDA 7.5. Set your build configuration to x64.
Then open the Visual Studio solution and build the projects.

### Build Documentation {#build_doc}
***

The Alea TK online documentation relies on Pandoc.

 1. Install Python version 3, which is used for the local web server
 1. Install [Pandoc](http://pandoc.org/) version 1.16 or later
 1. Install the Pandoc filter pandoc-eqnos to handle equation numbers
```
pip install pandoc-eqnos
```
 1. Check where **pandoc-eqnos.exe** is installed and add it to the path, usually it is in **Python Installation\\Python\\Scripts**

To build the documentation proceed as follows:
 
 1. Build the project **Generate** in the project folder **tutorial\\generate**.
 1. Open **Build.fsx**, select all, right-click and choose **Execute in Interactive** to execute the script in the F# interactive console. If you cannot find that window open it via **View &#8594; Other Windows &#8594; F# Interactive**.
 1. Go to **tutorial\\output** and start **run_server.bat** to launch a local web server. Open http://localhost:8080 in a browser to display the generated pages. 

### Adding New Tutorial Sample {#add_sample}
***

To add a new sample create a new project in **tutorial\\samples**. Add the required NuGet packages with **paket**

```
.paket\paket.exe add nuget Alea project <ProjectName>
.paket\paket.exe add nuget Alea.Parallel project <ProjectName>
```

Provide the meta information for the sample in **SampleProject.fs**:

```{.fsharp}
let metaData = 
    [
      {   
          Id = "0001"
          Title = "MNIST Digits Classificiation"
          Language = CSharp
          Tags = [ComputerVision; Regression; Classification; ConvolutionalNeuralNets]
          SourceCodeLink = sourceCodeLinkCsharp "MNIST"
          GitLink = gitLinkRoot + "MNIST"
          ImageLink = "images/gpu_device.svg"
          Src = "samples" @@ "MNIST"
      }
      
      // add here new meta data
      
  ]
```

We expect the following markdown files in the project folder:

  - Readme.md providing a short abstract of the samples
  - Extended.md containing detailed explanations
  
You can use LaTex formulas including formula numbering and reference the formulas. Check out an existing sample for more details. If the existing classification tags are not enough, you can add a new class. A button to select that class will be automatically generated. 

First rebuild the project **Generate** and then the documentation as explained above. 


