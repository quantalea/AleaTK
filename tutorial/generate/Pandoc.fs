module Pandoc

open System.IO
open Fake
open Util

// write to out file
let pandoc infile outfile template opt =
    let outfile = changeExt "html" outfile
    let cmd = 
        match template with
        | Some template -> sprintf "pandoc %s --template %s --mathjax --wrap=none --highlight-style tango -t html %s -o %s" infile template opt outfile
        | None -> sprintf "pandoc %s --mathjax --wrap=none --highlight-style tango -t html %s -o %s" infile opt outfile
    printfn "%s" cmd
    runCommand cmd
    
// returns html as string
let pandocString filename template opt =
    let cmd = 
        match template with
        | Some template -> sprintf "pandoc %s --template %s --mathjax --wrap=none --highlight-style tango -t html %s" filename template opt
        | None -> sprintf "pandoc %s --mathjax --wrap=none --highlight-style tango -t html %s" filename opt
    printfn "%s" cmd
    runCommandCaptureStdOut cmd
