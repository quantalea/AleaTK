# Documentation Tools

## Prerequisits

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

## Template Engine

We use [Templatus](https://github.com/kerams/Templatus), an F# based template enging similar to Microsoft's T4 engine 
to generate html code from *.tpl files.   
