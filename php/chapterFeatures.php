<?php
	header('Vary: Accept');
	define('FOLDER_BOOKS', 'books-txt/');
	define('FOLDER_GLOSSARIES', 'glossaries/');

	define('MIN_DETECT_CHARS', 1);
	define('RESET_BETWEEN_BOOKS', false);


	date_default_timezone_set('Europe/Zurich');

	include('_textutils.php');

	// =========================================================================================================================================================
	if (isset($_REQUEST['book'])) {
		header('Content-Type: text/plain; charset=utf-8');

		$parts = explode('.', $_REQUEST['book']);
		array_pop($parts);
		$test_keywords = FOLDER_GLOSSARIES.implode('.', $parts).'.json';
		if (@file_exists($test_keywords)) {
			$keywords = json_decode(file_get_contents($test_keywords),true);
			$sp = '([\s\.,&]|<p>)';
			$persos = array_keys($keywords);
		}

		$handle = @fopen(FOLDER_BOOKS.$_REQUEST['book'], 'r');
	#	db_x('TRUNCATE TABLE names;');
	#	db_x('TRUNCATE TABLE name_pages;');
		if ($handle) {
			$stats = array();
			$ln = 1;
			$book = 0;
			$last = false;
			$storySeenChars = array();
			$punct_signs = array('v'=>',', 'p'=>'.', 'pv'=>';', 'dp'=>':', 'tp'=>'…', 'pi'=>'?', 'pe'=>'!');
			while (($line = fgets($handle)) !== false) {
				str_replace('...', '…', $line);
				$parts = explode("\t", $line);
				if (count($parts) > 1) {
					list($book, $seqnum) = array_map('trim', explode('.', $parts[0]));
					$chapter_text = $parts[1];
					$phrases = preg_split('/[\.…;:\?\!]|\s-\s/', $chapter_text, NULL, PREG_SPLIT_NO_EMPTY);
					$stats[$book][$seqnum]['sent'] = count($phrases);
					$stats[$book][$seqnum]['words'] = count(preg_split('/\s/', $chapter_text, NULL, PREG_SPLIT_NO_EMPTY));
					$stats[$book][$seqnum]['sent_l'] = round(1000*$stats[$book][$seqnum]['words']/$stats[$book][$seqnum]['sent'])/1000;
					$phrasesStats = array();
					$punctuation = array();
					// Stylistic metrics
					foreach ($phrases as $phrase) {
						$words = preg_split('/\s/', $phrase);
						$phrasesStats[] = count($words);
					}
					$stats[$book][$seqnum]['phrase_len'] = array_sum($phrasesStats)/count($phrasesStats);
					$stats[$book][$seqnum]['phrase_len_stddev'] = stats_standard_deviation($phrasesStats);
					foreach ($punct_signs as $punct => $sign) {
						$stats[$book][$seqnum][$punct] = mb_substr_count($chapter_text, $sign);
					}
					// E/S persos
					if (RESET_BETWEEN_BOOKS && count($last)>0 && $book!=$last[0]) {
						$storySeenChars = array();
					}
					$stats[$book][$seqnum]['persos'] = array();
					foreach ($persos as $persoNamesStr) {
						$persoNames = array_map('trim', explode(',', $persoNamesStr));
						foreach ($persoNames as $perso) {
							$pc = mb_substr_count($chapter_text, $perso);
							if ($pc>=MIN_DETECT_CHARS) {
								@$stats[$book][$seqnum]['persos'][$persoNames[0]] += $pc;
								@$storySeenChars[$persoNames[0]] += 1;
							}
						}
					}
					$stats[$book][$seqnum]['seen'] = count(array_keys($storySeenChars));
					// Swap last/one to last and mark the concluding chapters
					if (RESET_BETWEEN_BOOKS) {
						$stats[$book][$seqnum]['pers_rm'] = 0;
					}
					$stats[$book][$seqnum]['last'] = $stats[$book][$seqnum]['onetol'] = 'n';
					if (count($last)>1 && ($book==$last[0] || !RESET_BETWEEN_BOOKS)) {
						if ($book == $last[0] || !RESET_BETWEEN_BOOKS) {
							$stats[$book][$seqnum]['pers_in'] = count(@array_diff(@array_keys($stats[$book][$seqnum]['persos']), @array_keys($stats[$book][$last[1]]['persos'])));
							$stats[$book][$seqnum]['pers_rm'] = count(@array_diff(@array_keys($stats[$book][$last[1]]['persos']), @array_keys($stats[$book][$seqnum]['persos'])));
							$stats[$book][$seqnum]['pers_k'] = count(@array_intersect(@array_keys($stats[$book][$seqnum]['persos']), @array_keys($stats[$book][$last[1]]['persos'])));
						}
						else {
							$stats[$last[0]][$last[1]]['last'] = 'y';
							$stats[$onetol[0]][$onetol[1]]['onetol'] = 'y';
						}
						$onetol = $last;
					}
					else {
						$stats[$book][$seqnum]['pers_in'] = count(@array_keys($stats[$book][$seqnum]['persos']));
					}
					$last = array($book, $seqnum);
				}
				else {
					echo 'invalid line at '.$ln.'<br/>';
				}
				$ln++;
			}
			$stats[$last[0]][$last[1]]['last'] = 'y';
			$stats[$onetol[0]][$onetol[1]]['onetol'] = 'y';

			$headers = array_merge(array('last', 'onetol', 'sent', 'pers_k', 'pers_in', 'pers_rm', 'seen', 'exit', 'ep', 'words', 'sent_l'), array_keys($punct_signs), array('phrase_len', 'phrase_len_stddev'));

			if (RESET_BETWEEN_BOOKS) {
				foreach ($stats as $book => $chapterStats) {
					$storySeenChars = array();
					$storyExitedChars = array();
					foreach ($chapterStats as $chapter => $values) {
						$storySeenChars = @array_unique(array_merge($storySeenChars, array_keys($values['persos'])));
					}
					foreach (array_reverse(array_keys($chapterStats)) as $i) {
						$storyExitedChars = @array_unique(array_merge($storyExitedChars, array_keys($chapterStats[$i]['persos'])));
						$stats[$book][$i]['exit'] = (count($storySeenChars) - count(@array_keys($storyExitedChars)));
					}
				}
			}
			else {
				$storySeenChars = array();
				$storyExitedChars = array();
				foreach ($stats as $book => $chapterStats) {
					foreach ($chapterStats as $chapter => $values) {
						$storySeenChars = @array_unique(array_merge($storySeenChars, array_keys($values['persos'])));
					}
				}
				$books = array_reverse(array_keys($stats));
				foreach ($books as $book) {
					$chapterStats = $stats[$book];
					foreach (array_reverse(array_keys($chapterStats)) as $i) {
						$storyExitedChars = @array_unique(array_merge($storyExitedChars, array_keys($chapterStats[$i]['persos'])));
						$stats[$book][$i]['exit'] = (count($storySeenChars) - count(@array_keys($storyExitedChars)));
					}
				}
			}
			echo "book\tchapter\t".implode("\t", array_values($headers))."\n";
			foreach ($stats as $book => $chapterStats) {
				foreach ($chapterStats as $c => $values) {
					echo $book."\t".$c;
					foreach ($headers as $h) {
						echo "\t".@$values[$h];
					}
					echo "\n";
				}
			}
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