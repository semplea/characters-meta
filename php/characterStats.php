<?php
	header('Vary: Accept');
	define('FOLDER_BOOKS', '../books-txt/');
	define('FOLDER_GLOSSARIES', '../glossaries/');

	define('MIN_DETECT_CHARS', 1);
	define('RESET_BETWEEN_BOOKS', false);


	date_default_timezone_set('Europe/Zurich');

	include('_textutils.php');

	// =========================================================================================================================================================
	if (isset($_REQUEST['book'])) {
		header('Content-Type: text/plain; charset=utf-8');
		$handle = @fopen(FOLDER_BOOKS.$_REQUEST['book'], 'r');
		if ($handle) {
			$np = array();
			$allWords = array();
			$ln = 1;
			while (($line = fgets($handle)) !== false) {
				$parts = explode("\t", mb_convert_encoding($line, 'UTF-8', mb_detect_encoding($line, 'auto', true)));
				if (count($parts) > 1) {
					$chap = $parts[0];
					$chapter_text = $parts[1];
					$chapter_phrases = array_values(array_filter(mb_split('[\.,;:\?\!«»]|\s-\s', $chapter_text)));
					foreach ($chapter_phrases as $p) {
						@$allWords[$chap].= $p.' ';
						$words = preg_split('/[\,\s\(\)\*]/', $p, NULL, PREG_SPLIT_NO_EMPTY);
						if (count($words)>2) {
							for ($i=1; $i<count($words)-1; $i++) {							// Ignore the first word (since it will be uppercase anyway)
								$w = $words[$i];
								if (mb_strlen($w)>1&&isUC($w)) {
									if (isUC($words[$i+1])||is_numeric($words[$i+1])) {
										$w = $w.' '.$words[$i+1];
										++$i;
									}
									@$np[$w]['count'] += 1;
									if (isset($words[$i-1])) {
										@$np[$w]['before'][] = $words[$i-1].' _';
									}
									if (isset($words[$i+1])) {
										@$np[$w]['after'][] = '_ '.$words[$i+1];
									}
								}
							}
						}
					}
				}
				else {
					echo 'invalid line at '.$ln.'<br/>';
				}
				$ln++;
			}

			mknatsort($np, array('count'), true);

			foreach ($allWords as $chap => $words) {
			}
/*
			foreach ($np as $n => $stats) {
				echo $n."\t";
				echo $stats['count']."\t";
				echo implode(',', array_unique($stats['after']))."\t";
				echo implode(',', array_unique($stats['before']))."\t";
				echo "\n";
			}
*/
			foreach ($np as $n => $stats) {
				$contextWords = (array)@array_merge(@array_unique($stats['before']), @array_unique($stats['after']));
				foreach ($contextWords as $w) {
					echo $n."\t".$w."\n";
				}
			}
	#		echo json_encode($np);
			fclose($handle);
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