<?php

	define('FOLDER_BOOKS', 'books-txt/');
	define('FOLDER_SPLIT', 'books-split/');
	define('FOLDER_GLOSSARIES', 'glossaries/');


	// =========================================================================================================================================================
	if (isset($_REQUEST['book'])) {
		header('Content-Type: text/plain; charset=utf-8');
		$bookRef = @array_shift(explode('.', $_REQUEST['book']));
		$folder = FOLDER_SPLIT.$bookRef;
		if (@mkdir($folder)) {
			chmod($folder, 0777);
		}
		$handle = @fopen(FOLDER_BOOKS.$_REQUEST['book'], 'r');
		if ($handle) {
			while (($line = fgets($handle)) !== false) {
				list($chapNum, $title, $text) = explode("\t", $line);
				list($tome, $chapter) = array_filter(explode('.', $chapNum));
				$file = FOLDER_SPLIT.$bookRef.'/'.$tome.'-'.str_pad($chapter, 2, '0', STR_PAD_LEFT).'.txt';
				$f = fopen($file, 'w');
		#		fwrite($f, '<h1>T. '.$c['tome'].'</h1>');
		#		fwrite($f, '<h2>§ '.$c['chapter'].'</h2>');
				$propertext = str_replace(' ', ' ', html_entity_decode(strip_tags(str_replace('</p>', "\n", $text))));
				$text = mb_convert_encoding($propertext, 'Windows-1252', 'UTF-8');
				fwrite($f, $propertext);
				chmod($file, 0777);
			}
		}
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