module Util

open System
open System.IO
open System.Text
open System.Diagnostics
open Fake

let joinStrings (sep:string) (s:seq<string>) = 
    String.Join(sep, s)

let lowerStr (a:'a) = (sprintf "%A" a).ToLower()

let normalizeName (name:string) = 
    name.Replace(' ', '_')
        .Replace('.', '_')
        .Replace(",", "")
        .Replace("#", "sharp")
        .ToLower()

let subDirectories path =
    let dInfo = new DirectoryInfo(path)
    dInfo.GetDirectories() |> Array.map (fun info -> info.FullName)

let ensureCommonSubDirectories pathOrigin pathTarget =
    subDirectories pathOrigin
    |> Array.map filename
    |> Array.map (fun subDir -> pathTarget @@ subDir)
    |> Array.iter ensureDirectory

let splitPath (path:string) =
    path.Split(Path.DirectorySeparatorChar)

let createTempDirectory() =
    let tempDirectory = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName())
    let dirInfo = Directory.CreateDirectory tempDirectory
    tempDirectory

let createTextFile (filename:string) (text:string) =
    use sw = File.CreateText filename
    sw.Write text
    sw.Close()

let runCommand cmd =
    let startInfo = new ProcessStartInfo()
    startInfo.FileName <- "cmd.exe"
    startInfo.WindowStyle <- ProcessWindowStyle.Normal
    startInfo.UseShellExecute <- false
    startInfo.Arguments <- @"/c " + cmd 
    use proc = Process.Start(startInfo)
    proc.WaitForExit()

let runCommandCaptureStdOut cmd =
    let startInfo = new ProcessStartInfo()
    startInfo.FileName <- "cmd.exe"
    startInfo.WindowStyle <- ProcessWindowStyle.Normal
    startInfo.UseShellExecute <- false
    startInfo.RedirectStandardOutput  <- true
    startInfo.Arguments <- @"/c " + cmd 

    use proc = Process.Start(startInfo)

    let builder = new StringBuilder()
    while not proc.HasExited do
        builder.Append(proc.StandardOutput.ReadToEnd()) |> ignore
    builder.Append(proc.StandardOutput.ReadToEnd()) |> ignore
    builder.ToString()

