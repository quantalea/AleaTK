#I @"..\..\packages\FAKE\tools"

#r "FakeLib.dll"

#load "Util.fs"
#load "Pandoc.fs"
#load "Types.fs"
#load "SampleProjects.fs"

open System
open System.Text
open System.IO
open System.Collections.Generic
open System.Diagnostics
open Fake
open Util
open Types
open Pandoc

Directory.SetCurrentDirectory __SOURCE_DIRECTORY__

let current = Directory.GetCurrentDirectory()
let tutorial = current @@ ".." 
let solution = tutorial @@ ".."
let design = tutorial @@ "design"
let output = tutorial @@ "output"
let doc = tutorial @@ "doc"
let templatus = solution @@ "packages" @@ "Templatus" @@ "tools" @@ "Templatus.exe"

Directory.SetCurrentDirectory tutorial

let copyToOutput folder =
    CopyDir (output @@ folder) (design @@ folder) allFiles  

let toHtmlString filename =
    let template = design @@ "templates" @@ "doc_page.html"
    pandocString filename (Some template)

let languages = ["csharp"; "fsharp"; "vb"]

let designFolders = ["content"; "fonts"; "images"; "scripts"]

let documents = 
    [
        "index.md"
        "get_started.md"
        "tutorials.md"
        "how_to.md"
        "ml_tools.md"
        "resources.md"
    ] 

// the directory structure for the sample documentation with separate tree for the extended documentation
let buildDirectoryStructure () =
    languages 
    |> List.iter (fun lang ->
        ensureDirectory (doc @@ "samples" @@ lang)
        ensureCommonSubDirectories (tutorial @@ "samples" @@ lang) (doc @@ "samples" @@ lang))

let ensureReadmeFiles () =
    languages
    |> List.iter (fun lang ->
        let ensure dir =
            subDirectories (dir @@ "samples" @@ lang)
            |> Array.iter (fun subDir -> CreateFile (subDir @@ "Readme.md"))
        ensure tutorial
        ensure doc)

let buildMainDoc clean =
    if clean then DeleteDirs [output]
    ensureDirectory output
    let runserver = output @@ "run_server.bat"
    if not (TestFile runserver) then
        createTextFile runserver "python -m http.server 8080"
    designFolders |> List.iter copyToOutput
    let template = design @@ "templates" @@ "doc_page.html"
    let bib = design @@ "templates" @@ "references.bib"
    let opts = sprintf "--filter pandoc-eqnos --ascii --bibliography %s" bib
    let toHtml filename = 
        let infile = tutorial @@ "doc" @@ filename
        let outfile = tutorial @@ "output" @@ filename
        pandoc infile outfile (Some template) opts
    documents |> List.iter toHtml

let buildSampleExtendedDocWithAbstract () =
    let template = tutorial @@ "generate" @@ "ExtendedSampleDoc.tpl"
    let bib = design @@ "templates" @@ "references.bib"
    let opts = sprintf "--filter pandoc-eqnos --ascii --bibliography %s" bib
    ensureDirectory output
    SampleProjects.metaData |> List.iter (fun meta -> 
        let outfile = tutorial @@ "output" @@ meta.ExtendedDocFile
        ensureDirectory (directory outfile)
        CopyFile outfile template
        let abstractDoc = tutorial @@ meta.Src @@ "Readme.md"
        let extendedDoc = tutorial @@ meta.Src @@ "Extended.md"
        let abstractHtml = Pandoc.pandocString abstractDoc None opts
        let extendedHtml = Pandoc.pandocString extendedDoc None opts
        // file is in samples, hence need to go up one directory
        ReplaceInFiles 
            [
                "$title$", meta.Title
                "$sourceCodeLink$", ("../" + meta.SourceCodeLink)
                "$gitLink$", ("../" + meta.SourceCodeLink)
                "$imageLink$", ("../" + meta.ImageLink)
                "$abstractHtml$", abstractHtml
                "$extendedHtml$", extendedHtml
            ] [outfile]
    )

let buildSampleGallery () =
    ensureDirectory output
    let tutorial = tutorial.Replace(@"\", @"\\") // this is a fix to properly pass '\' in parameters to Templatus

    let cmd = sprintf @" -t generate\SampleGallery.tpl -p tutorial=%s" tutorial
    runCommand (templatus + cmd)


