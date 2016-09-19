<a href="http://www.aleatk.com"><img border="0" alt="Alea TK" src="https://github.com/quantalea/AleaTK/blob/master/tutorial/design/images/Alea-TK-images.png" style="width:933px;">

***

<a href="http://www.aleatk.com">Alea TK</a> is an **open source** library for general purpose **numerical computing** and **Machine Learning** based on tensors and tensor expressions. 

- GPU accelerated
- Designed for rapid prototyping
- Developed in C# and usable from any .NET language

<a href="http://www.aleatk.com">http://www.aleatk.com</a>

## Package and building system

We use [Paket](http://fsprojects.github.io/Paket/index.html) to manage the packages used in this project, and a [FAKE](http://fsharp.github.io/FAKE/) script to build and publish. Here are some notes:

- Please always use `paket` to manage the packages, do not use the default NuGet package manager. This has many advantages, such as easier to make building script and publish.
- Some packages are pinned to certain version, for example, the NUnit is pinned to version 2 because of Resharper test runner. For more details, please check the `paket.dependencies` file in the solution root folder.
- When you made a fresh copy locally, you can always restore all packages by executing `.paket\paket.exe restore`. We also enabed the [package auto-restore](http://fsprojects.github.io/Paket/paket-auto-restore.html) feature, but if you created new project, you'd better enable that again for you new project.
- To add a package to your project, you could use `.paket\paket.exe add nuget YOUPACK version VERSION project YOUPROJ`, then commit the changed package management files.  