<?php
/*
	Need clean XML as input:
	tidy -asxhtml -bare -clean -utf8 -o /Users/cybor/Sites/3n/SHTotV8.htm -f errs.txt  /Users/cybor/Desktop/SHTotV8.htm

*/
	date_default_timezone_set('Europe/Zurich');
	mb_internal_encoding('UTF-8');

	define('FOLDER_BOOKS', 'books/');
	define('FOLDER_GLOSSARIES', 'glossaries/');

	// =========================================================================================================================================================
	function xslt($html, $xsl_file) {
		$xsl = new XSLTProcessor();
		$xsldoc = new DOMDocument();
		$xsldoc->load($xsl_file);
		$xsl->importStyleSheet($xsldoc);

		$xmldoc = new DOMDocument();
		$xmldoc->loadHTML($html);
		return $xsl->transformToXML($xmldoc);
	}

	// =========================================================================================================================================================
	if (isset($_REQUEST['book'])) {
		header('Content-Type: text/plain; charset=utf-8');
		$source = file_get_contents(FOLDER_BOOKS.$_REQUEST['book']);
	#	$source = preg_replace('/<html([^>]*)>/', '<html>', $source);
	#	$source = preg_replace('/<span style=\'mso-tab-count:.\'>\s+<\/span>/', '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;', $source);
	#	$source = str_replace(array('&nbsp;'), array(' '), $text);
		$source = preg_replace(array('/<span class="c21">&nbsp;<\/span>/', '/<p class="MsoNormal c([0-9]{2})"><\/p>/'), array('', '<p>&nbsp;</p>'), $source);
		$text = xslt($source, 'todb.xsl');
		$text = preg_replace(array('/\s+/u', '/<br clear="all" class=\'c[0-9][0-9]\' \/>/'), array(' ',''), $text);
		$chapter = 0;
		// Update formatting for import
		if (preg_match('/<h2>([0-9]+)\.([0-9]+)\.[\s]*/', $text) > 0) {
			$text = preg_replace('/<h2>([0-9]+)\.([0-9]+)\.[\s]*/', '<br/>==<br/>\1.\2. ', $text);
		}
		else {
			$i = 0;
			$text = preg_replace_callback('/<h2>/', function($matches) use (&$i) { return '<br/>==<br/>'.(++$i).'. '; }, $text);
		}
		$text = str_replace(array('</h2>', ' ', '>- ', ' :'), array('<br/>--<br/>', '&nbsp;', '>-&#8239;', '&nbsp;:'), $text);
		// Remove junk HTML tags
		$text = str_replace(array('<br/>', '<_>', '</_>', '<_/>', '<p/>', '<i/>'), array("\n"), $text);
		// Remove multiple line breaks
		$text = preg_replace('/[\n]+/u', "\n", $text);
		// Misc. fixes
		$text = str_replace(array('<?xml version="1.0" encoding="utf-8"?> <book xmlns="http://www.tei-c.org/ns/1.0">', '</book>', "--\n<p>&nbsp;</p>"), array('','',"--\n"), $text);
		// Keywords
		$parts = explode('.', $_REQUEST['book']);
		array_pop($parts);
		$test_keywords = FOLDER_GLOSSARIES.implode('.', $parts).'.json';
		if (@file_exists($test_keywords)) {
			$keywords = json_decode(file_get_contents($test_keywords),true);
			$sp = '([\s\.,&]|<p>)';
			foreach ($keywords as $word => $key) {
				$text = preg_replace('/'.$sp.$word.$sp.'/', '\1<a href="nomenclature.html#'.$key.'" class="nomenclature" data-word="'.$key.'">'.$word.'</a>\2', $text);
			}
		}
		// Punctuation rules
	#	$text = preg_replace('/\s?([\?\!])/u', ' $1', $text);
		$text = str_replace(array(' »', '« ', ' ?', ' !'), array('&#8239;»', '«&#8239;', '&#8239;?', '&#8239;!'), $text);

		echo $text.'==';
	}
	else {
		header('Content-Type: text/html; charset=utf-8');
		$books = scandir(FOLDER_BOOKS);
		foreach ($books as $_ => $book) {
			if (substr($book,0,1)!='.') {
				echo '<li><a href="?book='.$book.'">'.$book.'</a></li>';
			}
		}
	}

?>