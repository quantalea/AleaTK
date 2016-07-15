#I @"..\..\packages\FAKE\tools"
#load "HtmlBuilders.fsx"

open HtmlBuilders

// clean output
// DeleteDir output

buildMainDoc false
buildSampleExtendedDocWithAbstract () 
buildSampleGallery ()

