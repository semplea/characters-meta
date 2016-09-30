<?php

	include '../_sh.php';
	include '../_db/_db.php';

	$chapters = db_s('sequences', array(), array());
	while ($c = db_fetch($chapters)) {
		$file = $c['tome'].'-'.str_pad($c['chapter'], 2, '0', STR_PAD_LEFT).'.txt';
		$f = fopen($file, 'w');
#		fwrite($f, '<h1>T. '.$c['tome'].'</h1>');
#		fwrite($f, '<h2>§ '.$c['chapter'].'</h2>');
		$propertext = str_replace(' ', ' ', html_entity_decode(strip_tags(str_replace('</p>', "\n", $c['text']))));
		$text = mb_convert_encoding($propertext, 'Windows-1252', 'UTF-8');
		fwrite($f, $text);
		chmod($file, 0777);
	}
?>