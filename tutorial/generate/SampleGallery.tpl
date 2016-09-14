<#@ output filename="..\output\gallery.html" #>
<#@ assembly name=".\bin\Generate.dll" #>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">

    <head>
        <meta charset="utf-8">
        <title>Alea TK</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="Alea TK Documentation and Samples">
        <meta name="author" content="QuantAlea">
        <style type="text/css">
            code { white-space: pre; }
            q { quotes: "�" "�" "�" "�"; }
        </style>
        <link rel="shortcut icon" href="images/favicon.ico">

        <script src="https://code.jquery.com/jquery-2.1.4.min.js"></script>
        <script src="https://code.jquery.com/jquery-migrate-1.2.1.min.js"></script>
        <script type="text/javascript" src="scripts/jquery.vibrate.js"></script>
        <script type="text/javascript" src="scripts/jquery.quicksand.js"></script>
        <script type="text/javascript" src="scripts/jquery.powertip.js"></script>
        <link rel="stylesheet" type="text/css" href="scripts/jquery.powertip.css">

        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"></script>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css">
        <script src="scripts/bootbox.min.js"></script>

        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

        <script type="text/javascript" src="scripts/tips.js"></script>
        <script type="text/javascript" src="scripts/scripts.js"></script>
        <script type="text/javascript" src="scripts/version_list.js"></script>

        <link rel="stylesheet" type="text/css" href="content/src_highlight_tango.css">
        <link rel="stylesheet" type="text/css" href="content/gallery.css">
        <link rel="stylesheet" type="text/css" href="content/style.css">

        <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
        
    </head>

    <body>

        <div class="forkongithub" id="forkongithub">
            <a href="https://github.com/quantalea/AleaTK">Fork me on GitHub</a>
        </div>

        <div id="referenceContent" class="row">
            <div id="referenceContentInner">
                <div class="aside">
                    <div class="san_logo">
                        <a href="http://aleatk.com/">
                            <span data-picture="" data-alt="QuantAlea">
                            <span data-height="100" data-width="100" data-src="images/QuantAlea_cube_blau-grau_square_100.png"></span>
                            <span data-height="100" data-width="100" data-src="images/QuantAlea_cube_blau-grau_square_200.png" data-media="(-webkit-min-device-pixel-ratio:1.5), (min-resolution:1.5dppx)"></span>
                            <span data-height="100" data-width="100" data-src="images/QuantAlea_cube_blau-grau_square_200.png" data-media="(max-width:100px) and (-webkit-min-device-pixel-ratio:1.5), (max-width:1023px) and (min-resolution:144dpi)"><img alt="QuantAlea" width="100" src="images/QuantAlea_cube_blau-grau_square_200.png" height="100"></span>
                            <span data-src="images/QuantAlea_cube_blau-grau_square_100.png" data-media="(max-width:100px)"></span>
                            <span data-width="100" data-src="images/QuantAlea_cube_blau-grau_square_200.png" data-media="(max-width:100px) and (-webkit-min-device-pixel-ratio:1.5), (max-width:100) and (min-resolution:144dpi)"></span>
                            <noscript>&amp;amp;lt;img src="images/QuantAlea_cube_blau-grau_square.png" width="100" height="100" alt="Qcom"
                                &amp;amp;gt;
                            </noscript>
                            </span>
                        </a>
                    </div>
                    <ul class="nav nav-list" id="menu">
                        <li class="nav-header">Alea TK</li>
                        <li><a href="get_started.html">Get Started</a></li>
                        <li><a href="tutorials.html">Tutorials</a></li>
                        <li><a href="how_to.html">How To</a></li>
                        <li><a href="ml_tools.html">Machine Learning Tools</a></li>
                        <li><a href="resources.html">Resources</a></li>
                        <li><a href="design_details.html">Design Details</a></li>
                        <li><a href="gallery.html">Sample Gallery</a></li>
                        <li class="nav-header">QuantAlea</li>
                        <li><a href="http://quantalea.com/">Web</a></li>
                        <li><a href="http://blog.quantalea.com">Blog</a></li>
                        <li><a href="http://www.aleagpu.com/release/">Alea GPU</a></li>

                        <div id="version_list">Loading versions ...</div>
                        <!--<div class="dropdown">
                          <button class="dropbtn">Version</button>
                          <div class="dropdown-content">
                            <a href="http://www.aleatk.com/release/0_9_0/doc/">0.9.0</a>
                          </div>
                        </div>-->
                    </ul>
                </div>

                <div class="center" id="main">

                    <h2>Sample Gallery</h2>

                    <div class="container">
                        <article>

                            <div id="samples">
							    <!-- Languages -->
								<!-- for now we have just one language
								<ul class="lang-filter nav nav-pills gallery">
									<h2>Languages</h2>
									<#
									    pushIndent "                                    "
									    SampleProjects.languages
									    |> Seq.iter	(fun (lang, text) ->
									    	tprintfn """<li data-value="%s"><button type="button" class="btn btn-default btn-sm">%s</button></li>""" (Util.lowerStr lang) text
									    )
									    popIndent ()
									#>
								</ul> -->

                                <!-- Categories -->
                                <div class="row">
                                  <div class="col-md-11">
                                    <ul class="filter nav nav-pills">
                                        <li data-value="all"><button id="all" type="button" class="btn btn-default btn-sm">All</button></li>
                                            <#
                                            	pushIndent "                                    "
                                            	SampleProjects.tags
                                            	|> Seq.iter	(fun (tag, text) ->
                                            		tprintfn """<li data-value="%s"><button type="button" class="btn btn-default btn-sm">%s</button></li>""" (Util.lowerStr tag) text
                                            	)
                                            	popIndent ()
                                            #>
                                    </ul>
                                  </div>
                                </div>

                                <div class="row">
                                  <div class="col-md-11">
                                    <ul class="thumbnails">
                                        <!-- BEGIN THUMBNAILS : will be auto generaed from script -->
                                        <#
                                            pushIndent "                                    "
                                            SampleProjects.metaData
                                            |> Seq.iteri (fun i meta ->
                                                let tags =
                                                    meta.Tags
                                                    |> Seq.map Util.lowerStr
                                                    |> Util.joinStrings " "
												// for now we just have one language
                                                // tprintfn """<li lang-type="%s" data-type="%s" data-id="id-%d" class="sample-app">""" (Util.lowerStr meta.Language) tags i
                                                tprintfn """<li lang-type="csharp" data-type="%s" data-id="id-%d" class="sample-app">""" tags i
                                                tprintfn """   <a href="#" id="thumb-%d"><h2>%s</h2><img src="%s" alt="%s"></a>""" i meta.Title meta.ImageLink meta.Title
                                                tprintfn """</li>""")
                                            popIndent ()
                                        #>
                                        <!-- END THUMBNAILS -->
                                    </ul>
                                </div>
                              </div>

                                <!-- BEGIN TEXT : will be auto generaed from script -->

                              <#
                                  pushIndent "                                "
                                  SampleProjects.metaData
                                  |> Seq.iteri (fun i meta ->
                                      let filename = sprintf "%s\\%s" tutorial meta.Abstract
                                      printfn "converting to html : %s" filename
                                      let html = Pandoc.pandocString filename None ""
                                      tprintfn """<div id="text-%d" class="hidden">""" i
                                      tprintfn """    <p>%s</p> """ html
                                      tprintfn """</div>""")
                                  popIndent ()
                              #>


                                <!-- END TEXT -->
                            </div>

                            <script>
                                function gallery() {
                                // BEGIN JAVASCRIPT : will be auto generaed from script
                                <#
                                	let span = "                                    "
                                	SampleProjects.metaData
                                	|> Seq.iteri (fun i meta ->
                                        tprintfn """%s$('#thumb-%d').off().on('click', function () {""" span i
                                        tprintfn """%s    var content = $('#text-%d').clone(false);""" span i
                                        tprintfn """%s    $(content).removeClass('hidden');""" span
                                        tprintfn """%s    bootbox.dialog({ message: content, title: '%s', buttons: {""" span meta.Title
                                        tprintfn """%s            "Read more": { className: 'btn-primary', callback: function() { window.open('%s', '_blank'); }},""" span meta.ExtendedDocLink
                                        tprintfn """%s            "Checkout the code": {className: 'btn-primary', callback: function() { window.open('%s', '_blank'); }},""" span meta.GitLink
                                        tprintfn """%s            "Download" : {className: 'btn-primary', callback: function() { window.open('%s', '_blank'); }}""" span meta.SourceCodeLink
                                        tprintfn """%s        }});""" span
                                        tprintfn """%s});""" span)
                                #>
                                // END JAVASCRIPT
                                }

                                var itemsHolder = $('ul.thumbnails');
                                var items = itemsHolder.clone();
                                // Initially list all items for the default language (csharp)
                                var selectedLanguage = 'csharp';
                                // Container for all active selection tags
                                var activeTags = [];
                                var filtered = items.find('li[lang-type="' + selectedLanguage + '"]');
                                $("li[data-value='csharp']").find('button').addClass('btn-primary');
                                $('#all').addClass('btn-primary');
                                itemsHolder.quicksand(filtered, { duration: 1000 }, gallery);

                                // Filter based on language change
                                $('ul.lang-filter li').click(function (e) {
                                    e.preventDefault();
                                    selectedLanguage = $(this).attr('data-value');

                                    $('.lang-filter').find('button').removeClass('btn-primary');
                                    $(this).find("button").addClass('btn-primary');

                                    filtered = items.find('li[lang-type=' + selectedLanguage + ']');
                                    itemsHolder.quicksand(filtered, { duration: 1000 }, gallery);
                                });


                                $('ul.filter li').click(function (e) {
                                    e.preventDefault();

                                    // Determine the tag type (e.g basic vs parralel-for, etc)
                                    var selectedTag = $(this).attr('data-value');

                                    // Display all items in case we are listing all
                                    if (selectedTag == 'all') {
                                        // reset style for all other filter tags (as none are selected)
                                        $('.filter').find('button').removeClass('btn-primary');
                                        $(this).find("button").addClass('btn-primary');
                                        activeTags = [];
                                        filtered = items.find('li[lang-type=' + selectedLanguage + ']');
                                    } else {
                                      // Remove the selected state of the "ALL" button
                                      $('#all').removeClass('btn-primary');
                                      // Save the selected element to the filter array if it hasn't been added already
                                      if (activeTags.indexOf(selectedTag) === -1) {
                                        activeTags.push(selectedTag);
                                        $(this).find("button").addClass('btn-primary');
                                      } else {
                                        // Remove the element and restore the deselected state
                                        activeTags.splice(activeTags.indexOf(selectedTag), 1 );
                                        $(this).find("button").removeClass('btn-primary');
                                      }
                                      filtered = [];
                                      // Filter based on the language type and the selected tags
                                      for (var i = 0; i < activeTags.length; i++) {
                                        var matches = items.find('li[data-type~=' + activeTags[i] + '][lang-type=' + selectedLanguage + ']');
                                        $.merge(filtered, matches);
                                      }
                                    }

                                    itemsHolder.quicksand(filtered, { duration: 1000 }, gallery);
                                });

                                window.onresize = function(event) {
                                    items.removeAttr('style');
                                };

                                $(document).ready(gallery);
                            </script>

                        </article>
                    </div>
                </div>
            </div>
        </div>
    </body>
	<script type="text/javascript" src="https://www.draw.io/embed.js?s=arrows"></script>
</html>
