#I "packages/FAKE/tools"
#r "FakeLib.dll"
#load "tutorial/generate/HtmlBuilders.fsx"

open System
open System.IO
open System.Text
open System.Text.RegularExpressions
open System.Xml
open Fake
open Fake.AssemblyInfoFile

Directory.SetCurrentDirectory __SOURCE_DIRECTORY__

// Check versions before making release
// e.x: Some "beta", None (for stable)
let versionType = Some "beta"
let majorVersion = 0
let minorVersion = 9
let patchVersion = 0
let buildVersion = 4
// End of version section

let version = sprintf "%d.%d.%d.%d" majorVersion minorVersion patchVersion buildVersion
let nugetVersion =
    match versionType with
    | None -> sprintf "%d.%d.%d" majorVersion minorVersion patchVersion
    | Some t -> sprintf "%d.%d.%d-%s%d" majorVersion minorVersion patchVersion t buildVersion
let versionSubDir = nugetVersion.Replace('.', '_')
let thisTarget = environVar "target"
let now = DateTime.Now

let mutable ParamDoClean          = not (hasBuildParam "--no-clean")
let mutable ParamDoBuild          = not (hasBuildParam "--no-build")
let mutable ParamDoDoc            = not (hasBuildParam "--no-doc")
let mutable ParamDoTest           = not (hasBuildParam "--no-test")
let mutable ParamDoPackage        = not (hasBuildParam "--no-package")
let mutable ParamDoPublish        = thisTarget = "Publish"
let mutable ParamShowTest         = hasBuildParam "--show-test"
let mutable ParamIgnoreTestFail   = hasBuildParam "--ignore-test-fail"
let mutable ParamIncludePaket     = hasBuildParam "--include-paket"
let mutable ParamAddLocalSource   = hasBuildParam "--add-local-source"

// if publish, all tests must pass
ParamDoTest <- if ParamDoPublish then true else ParamDoTest
ParamIgnoreTestFail <- if ParamDoPublish then false else ParamIgnoreTestFail
ParamIncludePaket <- if ParamDoPublish then false else ParamIncludePaket
ParamAddLocalSource <- if ParamDoPublish then false else ParamAddLocalSource

let product = "Alea TK"
let company = "QuantAlea AG."
let copyright = sprintf "QuantAlea AG. 2016-%d" now.Year
let projectUrl = "http://www.aleatk.com"
let iconUrl = "http://quantalea.com/static/app/images/favicon256x256.ico"
let licenseUrl = "http://www.aleatk.com/license"
let releaseNotes = "http://www.aleatk.com/releasenotes"
let nugetExePath = "packages" @@ "NuGet.CommandLine" @@ "tools" @@ "NuGet.exe"

printfn "===================================="
printfn "Now                    : %A" now
printfn "DoClean                : %A" ParamDoClean
printfn "DoBuild                : %A" ParamDoBuild
printfn "DoDoc                  : %A" ParamDoDoc
printfn "DoTest                 : %A" ParamDoTest
printfn "DoPackage              : %A" ParamDoPackage
printfn "DoPublish              : %A" ParamDoPublish
printfn "ShowTest               : %A" ParamShowTest
printfn "IgnoreTestFail         : %A" ParamIgnoreTestFail
printfn "IncludePaket           : %A" ParamIncludePaket
printfn "ParamAddLocalSource    : %A" ParamAddLocalSource
printfn "Version                : %s" version
printfn "VersionType            : %s" (match versionType with Some t -> t | None -> "[stable]")
printfn "NuGetVersion           : %s" nugetVersion
printfn "===================================="

[<Literal>]
let nuspecTemplate = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <metadata minClientVersion="2.8" xmlns="http://schemas.microsoft.com/packaging/2010/07/nuspec.xsd">    
    <id>@project@</id>
    <version>@build.number@</version>
    <authors>@authors@</authors>
    <owners>@authors@</owners>
    <developmentDependency>__DEVELOPMENT_DEPENDENCY__</developmentDependency>
    <summary />
    __LICENSE_URL__
    <projectUrl>__PROJECT_URL__</projectUrl>
    <iconUrl>__ICON_URL__</iconUrl>
    <requireLicenseAcceptance>__REQUIRE_LICENSE_ACCEPTANCE__</requireLicenseAcceptance>
    <description>@description@</description>
    <releaseNotes>@releaseNotes@</releaseNotes>
    <copyright>@copyright@</copyright>    
    <tags>@tags@</tags>
    @dependencies@
  </metadata>
</package>"""

let writeNuSpecFile path (projectUrl:string) (iconUrl:string) licenseUrl devOnlyDep =
    let template = nuspecTemplate
    let template = Regex.Replace(template, "__PROJECT_URL__", projectUrl)
    let template = Regex.Replace(template, "__ICON_URL__", iconUrl)
    let template = licenseUrl |> function
        | None ->
            let template = Regex.Replace(template, "__LICENSE_URL__", "")
            Regex.Replace(template, "__REQUIRE_LICENSE_ACCEPTANCE__", "false")
        | Some licenseUrl ->
            let template = Regex.Replace(template, "__LICENSE_URL__", sprintf "<licenseUrl>%s</licenseUrl>" licenseUrl)
            Regex.Replace(template, "__REQUIRE_LICENSE_ACCEPTANCE__", "true")
    let template = Regex.Replace(template, "__DEVELOPMENT_DEPENDENCY__", sprintf "%A" devOnlyDep)
    File.WriteAllText(path, template)

[<Literal>]
let slnTemplate = """Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio 14
VisualStudioVersion = 14.0.25123.0
MinimumVisualStudioVersion = 10.0.40219.1
Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "__PROJECT_NAME__", "tutorial\samples\__PROJECT_NAME__\__PROJECT_NAME__.csproj", "__PROJECT_GUID__"
EndProject
Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "AleaTKUtil", "src\AleaTKUtil\AleaTKUtil.csproj", "{19810D0A-BD63-4360-ACA3-1DB47C91E0A3}"
EndProject
Global
    GlobalSection(SolutionConfigurationPlatforms) = preSolution
        Debug|Any CPU = Debug|Any CPU
        Release|Any CPU = Release|Any CPU
    EndGlobalSection
    GlobalSection(ProjectConfigurationPlatforms) = postSolution
        __PROJECT_GUID__.Debug|Any CPU.ActiveCfg = Debug|Any CPU
        __PROJECT_GUID__.Debug|Any CPU.Build.0 = Debug|Any CPU
        __PROJECT_GUID__.Release|Any CPU.ActiveCfg = Release|Any CPU
        __PROJECT_GUID__.Release|Any CPU.Build.0 = Release|Any CPU
		{19810D0A-BD63-4360-ACA3-1DB47C91E0A3}.Debug|Any CPU.ActiveCfg = Debug|Any CPU
		{19810D0A-BD63-4360-ACA3-1DB47C91E0A3}.Debug|Any CPU.Build.0 = Debug|Any CPU
		{19810D0A-BD63-4360-ACA3-1DB47C91E0A3}.Release|Any CPU.ActiveCfg = Release|Any CPU
		{19810D0A-BD63-4360-ACA3-1DB47C91E0A3}.Release|Any CPU.Build.0 = Release|Any CPU
    EndGlobalSection    
    GlobalSection(SolutionProperties) = preSolution
        HideSolutionNode = FALSE
    EndGlobalSection
EndGlobal"""  

let writeSolutionFile path (projectName:string) (projectGuid:string) = 
    let template = slnTemplate 
    let template = Regex.Replace(template, "__PROJECT_NAME__", projectName)
    let template = Regex.Replace(template, "__PROJECT_GUID__", projectGuid)
    File.WriteAllText(path, template)

Target "Clean" (fun _ ->
    CreateDir "release"
    CreateDir "tutorial/output"
    CreateDir "temp"
    CreateDir "output"
    "temp" :: [] |> CleanDirs
    if ParamDoClean then "output" :: "release" :: "tutorial/output" :: [] |> CleanDirs
)

Target "Build" (fun _ ->
    if ParamDoBuild then

        // update solution info file
        let assemblyInfos =
            [ Attribute.Product product
              Attribute.Company company
              Attribute.Copyright copyright
              Attribute.Version version
              Attribute.FileVersion version
              Attribute.InformationalVersion version ]

        CreateCSharpAssemblyInfo "SolutionInfo.cs" assemblyInfos
        CreateFSharpAssemblyInfo "SolutionInfo.fs" assemblyInfos

        // build
        build (fun p ->
            { p with
                Verbosity = Some Minimal
                MaxCpuCount = Some (Some 1)
                Targets = "Clean" :: "Build" :: []
                Properties =
                    [
                        "Optimize", "True"
                        "Configuration", "Release"
                    ]
            } ) "AleaTK.sln"
        |> DoNothing
)

Target "Doc" (fun _ ->
    if ParamDoDoc then
        Directory.SetCurrentDirectory (__SOURCE_DIRECTORY__ @@ "tutorial")
        HtmlBuilders.buildMainDoc false
        HtmlBuilders.buildSampleExtendedDocWithAbstract () 
        HtmlBuilders.buildSampleGallery ()
        Directory.SetCurrentDirectory __SOURCE_DIRECTORY__
)

Target "Test" (fun _ ->
    if ParamDoTest then
        let ignoreTest part name = sprintf "Ignore %s Test => %s" part name |> traceImportant; None
        let tests = "AleaTKTest.exe" :: []
        let samples = 
            !! "tutorial/samples/*/*.csproj"
            ++ "tutorial/samples/*/*.fsproj"
            |> Seq.map Path.GetFileNameWithoutExtension

            // filter out these samples, because they need download dataset and that is too long
            |> Seq.choose (fun name ->
                match name with
                | "MNIST" -> ignoreTest "Sample" name
                | "PTB" -> ignoreTest "Sample" name
                | "WMT" -> ignoreTest "Sample" name
                | name when (File.Exists("release" @@ (sprintf "%s.exe" name))) -> sprintf "%s.exe" name |> Some
                | name when (File.Exists("release" @@ (sprintf "%s.dll" name))) -> sprintf "%s.dll" name |> Some
                | name -> ignoreTest "Sample" name )
            
            |> Seq.toList

        let tests = tests :: samples :: [] |> List.concat
        
        tests |> List.iter (printfn "Test => %s")
        tests |> NUnit (fun p ->
            { p with
                ToolPath = "packages/NUnit.Runners.Net4/tools"
                OutputFile = "TestResults.xml"
                Out = if ParamShowTest then "" else "TestStdOut.txt"
                ErrorOutputFile = if ParamShowTest then "" else "TestErrorOut.txt"
                TimeOut = TimeSpan.FromHours 2.0
                Framework = "net-4.0"
                ShowLabels = ParamShowTest
                StopOnError = not ParamIgnoreTestFail
                DisableShadowCopy = true
                ProcessModel = NUnitProcessModel.MultipleProcessModel
                Domain =  NUnitDomainModel.DefaultDomainModel
                WorkingDir = "release"
            } )

        CopyFile "output" "release/TestResults.xml"
        if not ParamShowTest then
            !! "TestStdOut.txt"
            ++ "TestErrorOut.txt"
            |> FileSystem.SetBaseDir "release"
            |> CopyFiles "output"
)

Target "Package" (fun _ ->
    if ParamDoPackage then
        let tempDir = "temp"
        //let releaseOutputDir = ("output" @@ (sprintf "AleaTK.Release.%s" nugetVersion) @@ versionSubDir)
        let releaseOutputDir = ("output" @@ "release" @@ versionSubDir)
        let equalVersion (version:string) = sprintf "[%s]" version

        let package_AleaTK() =
            let tempDir = tempDir @@ "package_AleaTK"
            CleanDir tempDir

            let project = "AleaTK"
            let description ="Library for general purpose numerical computing and Machine Learning based on tensors and tensor expressions." 
            let tags = "Alea GPU CUDA QuantAlea Machine Learning"
            writeNuSpecFile (tempDir @@ "package.nuspec") projectUrl iconUrl (Some licenseUrl) false

            let rootDir = tempDir @@ "files"
            CreateDir rootDir

            let libDir = rootDir @@ "lib" @@ "net40"
            CreateDir libDir
            !! "AleaTK.dll"
            ++ "AleaTK.xml"
            |> FileSystem.SetBaseDir "release"
            |> CopyFiles libDir

            let dependencies =
                [ "Alea"                , GetPackageVersion "packages" "Alea" |> equalVersion ]

            NuGet (fun p ->
                { p with
                    ToolPath = nugetExePath
                    Project = project
                    Version = nugetVersion
                    Authors = company :: []
                    Description = description
                    ReleaseNotes = releaseNotes
                    Dependencies = dependencies
                    Copyright = copyright
                    Tags = tags
                    OutputPath = "output"
                    WorkingDir = rootDir })
                    (tempDir @@ "package.nuspec")

        let deploySamples() =
            let cwd = Directory.GetCurrentDirectory()
            let sampleOutputFolder = (releaseOutputDir @@ "samples")
            CreateDir sampleOutputFolder
            CleanDir sampleOutputFolder

            !! "tutorial/samples/*/*.csproj"
            ++ "tutorial/samples/*/*.fsproj"

            |> Seq.filter (fun projectFile ->
                let projectName = Path.GetFileNameWithoutExtension(projectFile)
                match projectName with
                | _ -> true)

            |> Seq.iter (fun projectFile ->
                let projectName = Path.GetFileNameWithoutExtension(projectFile)
                let projectFolder = Path.GetDirectoryName(projectFile)

                let tempRootFolder = "temp" @@ projectName |> Path.GetFullPath
                let tempProjectFolder = tempRootFolder @@ (projectFolder.Substring(cwd.Length + 1))
                let tempProjectFile = tempProjectFolder @@ (Path.GetFileName(projectFile))

                // copy AleaTKUtil
                CopyDir (tempRootFolder @@ "src" @@ "AleaTKUtil") ("src" @@ "AleaTKUtil") (fun file -> true)
                DeleteDir (tempRootFolder @@ "src" @@ "AleaTKUtil" @@ "bin")
                DeleteDir (tempRootFolder @@ "src" @@ "AleaTKUtil" @@ "obj")
        
                // copy the project files, then delete the intermediate folders
                CopyDir tempProjectFolder projectFolder (fun file -> true)
                DeleteDir (tempProjectFolder @@ "bin")
                DeleteDir (tempProjectFolder @@ "obj")

                // copy the paket binaries
                CreateDir (tempRootFolder @@ ".paket")
                CopyFile (tempRootFolder @@ ".paket") ".paket/paket.bootstrapper.exe"
                if ParamIncludePaket then CopyFile (tempRootFolder @@ ".paket") ".paket/paket.exe"

                // copy the paket lock file
                CopyFile tempRootFolder "paket.lock"

                // copy dependencies file, add source
                let lines = File.ReadAllLines("paket.dependencies")
                let sb = StringBuilder()
                for line in lines do
                    let line = line.Trim()
                    if (ParamAddLocalSource) && (line = "source https://www.nuget.org/api/v2") then
                        sb.AppendLine(sprintf "source %s" (Path.GetFullPath(".") @@ "output"))
                            .AppendLine(line)
                            |> ignore
                    else sb.AppendLine(line) |> ignore
                File.WriteAllText(tempRootFolder @@ "paket.dependencies", sb.ToString())

                // create script to restore and add alea packages for sample project
                let sb = StringBuilder()
                if not ParamIncludePaket then sb.AppendLine(".paket\paket.bootstrapper.exe") |> ignore
                sb.AppendLine(sprintf ".paket\paket.exe restore")
                    .AppendLine(sprintf ".paket\paket.exe add nuget AleaTK version %s project %s" nugetVersion (Path.GetFileNameWithoutExtension(projectFile)))
                    |> ignore
                File.WriteAllText(tempRootFolder @@ "paket_setup.bat", sb.ToString())

                // create a readme instruction file
                let sb = StringBuilder()
                sb.AppendLine("## Steps to build and run samples")
                    .AppendLine("1. Double click `paket_setup.bat` to restore/install packages via [Paket](https://fsprojects.github.io/Paket/index.html)")
                    .AppendLine("2. Open solution file with Visual Studio 2015")
                    .AppendLine("3. Build solution")
                    .AppendLine("4. For an application sample, press Ctrl-F5 to run it, for nunit tests, use a test runner to run tests")
                    |> ignore
                File.WriteAllText(tempRootFolder @@ "readme.md", sb.ToString())

                // delete the project reference element in the project file
                let doc = XmlDocument()
                doc.Load(projectFile)

                let mgr = XmlNamespaceManager(doc.NameTable)
                mgr.AddNamespace("x", "http://schemas.microsoft.com/developer/msbuild/2003")

                // write solution file using the project guid
                let nodeProjectGuidList = doc.SelectNodes("//x:PropertyGroup/x:ProjectGuid", mgr)
                if nodeProjectGuidList.Count > 0 then   
                    let guid = nodeProjectGuidList.Item(0).InnerText
                    let solutionFile = projectName + ".sln"
                    printfn "Writing solution file %s" solutionFile
                    writeSolutionFile (tempRootFolder @@ solutionFile) projectName (sprintf "%s" guid)
                else 
                    failwithf "could not read project GUID from %s" projectFile

                for nodeProjectReferenceName in doc.SelectNodes("//x:ItemGroup/x:ProjectReference/x:Name", mgr) do
                    let name = nodeProjectReferenceName.InnerText
                    match name with
                    | "AleaTK" ->
                        printfn "Removing ProjectReference %A" name
                        let nodeProjectReference = nodeProjectReferenceName.ParentNode
                        let nodeItemGroup = nodeProjectReference.ParentNode
                        nodeItemGroup.RemoveChild(nodeProjectReference) |> ignore

                    | projectReferenceName ->
                        printfn "Skipping ProjectReference %A" projectReferenceName 

                doc.Save(tempProjectFile)

                // now zip it and put it in the output folder
                !! (sprintf "%s/**/*" tempRootFolder)
                |> Zip tempRootFolder (sampleOutputFolder @@ (sprintf "%s.zip" projectName))
            )

        let deployDoc() =
            let cwd = Directory.GetCurrentDirectory()
            let docOutputFolder = (releaseOutputDir @@ "doc")
            CreateDir docOutputFolder
            CleanDir docOutputFolder

            CopyDir docOutputFolder "tutorial/output" (fun file -> true)
            DeleteFile (docOutputFolder @@ "run_server.bat")

            "index.html" |> CopyFile ("output" @@ "release")
            "version_list.html" |> CopyFile ("output" @@ "release")
            ("tutorial" @@ "output" @@ "run_server.bat") |> CopyFile "output"

        package_AleaTK() 
        deploySamples()
        deployDoc()
)

Target "Publish" (fun _ ->
    let mutable pushed = 0

    let pushNupkg (name:string) =
        let filename = sprintf "%s.%s.nupkg" name nugetVersion
        let filepath = "output" @@ filename

        if 0 = ExecProcess(fun info ->
            info.FileName <- nugetExePath
            info.WorkingDirectory <- null
            info.Arguments <- sprintf "push %s %s -Timeout 3600 -Verbosity detailed -NonInteractive" filepath (Environment.GetEnvironmentVariable("QUANTALEA_NUGET_KEY")))
            (TimeSpan.FromHours 2.)
        then pushed <- pushed + 1
        else sprintf "Push %s to nuget.org failed." filename |> traceImportant

    pushNupkg "AleaTK"
)

Target "Help" (fun _ ->
    traceImportant "[Targets]"
    traceImportant "  Clean => Build => Test => Package"
    traceImportant "[Folders]"
    traceImportant "  release : for build output"
    traceImportant "  temp    : for temp"
    traceImportant "  output  : nuget package and zip files"
    traceImportant "[Options]"
    traceImportant "  --no-clean           : do not clean release folder"
    traceImportant "  --no-build           : skip build"
    traceImportant "  --no-doc             : skip doc"
    traceImportant "  --no-test            : skip test"
    traceImportant "  --no-package         : skip package"
    traceImportant "  --show-test          : show test output while testing"
    traceImportant "  --ignore-test-fail   : ignore failed test and continue"
    traceImportant "  --include-paket      : include paket.exe in sample, save time for developing"
    traceImportant "[Examples]"
    traceImportant "  ] build.bat Package (--no-test) (--include-paket)"
)

"Clean" 
    ==> "Build"
    ==> "Doc"
    ==> "Test"
    ==> "Package"
    ==> "Publish"

RunTargetOrDefault "Help"
