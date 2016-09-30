<?php


	function array_flatten(array $a) {
		$n = array();
		array_walk_recursive($a, function($_a) use (&$n) { $n[] = $_a; });
		return $n;
	}
	function isLC($w) {
		$fl = ord(substr($w, 0, 1));
		return ($fl > ord('Z') && $fl <= ord('z'));
	}
	function isUC($w) {
		$fl = ord(substr($w, 0, 1));
		return ($w != '' && $fl >= ord('A') && $fl <= ord('Z'));
	}
	function part_join($w1, $w2) {
		$sep = (strlen($w1)==0||substr($w1, -1)=="'"?'':' ');				// No space if simple quote in the middle
		return $w1.$sep.$w2;
	}
	function words_implode($words) {
		$wString=array_shift($words);
		foreach($words as $w) {
			if (substr($wString, -1)!="'" && $w!='') {
				$wString.=' ';
			}
			$wString.=$w;
		}
		return trim($wString);
	}
	function unsorted($a2) {
		$a1 = array();
		foreach ($a2 as $idx=>$v) {
			$a1[] = $v;
		}
		sort($a1, SORT_NUMERIC);
		for ($i=0; $i<count($a1); $i++) {
			if (intval($a1[$i]) != intval($a2[$i])) {
				return $a2[$i].'/'.$i.' (expected '.$a1[$i].')';
			}
		}
		return false;
	}

	if (!function_exists('stats_standard_deviation')) {
		/**
		* This user-land implementation follows the implementation quite strictly;
		* it does not attempt to improve the code or algorithm in any way. It will
		* raise a warning if you have fewer than 2 values in your array, just like
		* the extension does (although as an E_USER_WARNING, not E_WARNING).
		* http://php.net/manual/en/function.stats-standard-deviation.php
		*
		* @param array $a
		* @param bool $sample [optional] Defaults to false
		* @return float|bool The standard deviation or false on error.
		*/
		function stats_standard_deviation(array $a, $sample = false) {
			$n = count($a);
			if ($n === 0) {
				trigger_error("The array has zero elements", E_USER_WARNING);
				return false;
			}
			if ($sample && $n === 1) {
				trigger_error("The array has only 1 element", E_USER_WARNING);
				return false;
			}
			$mean = array_sum($a) / $n;
			$carry = 0.0;
			foreach ($a as $val) {
				$d = ((double) $val) - $mean;
				$carry += $d * $d;
			};
			if ($sample) {
				--$n;
			}
			return sqrt($carry / $n);
		}
	}

	function getWiki($s) {
	    $url = "http://fr.wikipedia.org/w/api.php?action=opensearch&search=".urlencode($s)."&format=json&limit=1";
	    $ch = curl_init($url);
	    curl_setopt($ch, CURLOPT_HTTPGET, TRUE);
	    curl_setopt($ch, CURLOPT_POST, FALSE);
	    curl_setopt($ch, CURLOPT_HEADER, false);
	    curl_setopt($ch, CURLOPT_NOBODY, FALSE);
	    curl_setopt($ch, CURLOPT_VERBOSE, FALSE);
	    curl_setopt($ch, CURLOPT_REFERER, "");
	    curl_setopt($ch, CURLOPT_FOLLOWLOCATION, TRUE);
	    curl_setopt($ch, CURLOPT_MAXREDIRS, 4);
	    curl_setopt($ch, CURLOPT_RETURNTRANSFER, TRUE);
	    curl_setopt($ch, CURLOPT_USERAGENT, "Mozilla/5.0 (Windows; U; Windows NT 6.1; he; rv:1.9.2.8) Gecko/20100722 Firefox/3.6.8");
	    $page = curl_exec($ch);
	    curl_close($ch);
	    $xml = simplexml_load_string($page);
	    print_r($xml);
	    if((string)$xml->Section->Item->Description) {
	        return array((string)$xml->Section->Item->Text, (string)$xml->Section->Item->Description, (string)$xml->Section->Item->Image['source'], (string)$xml->Section->Item->Url);
	    } else {
	        return array();
	    }
	}

	/**
	*    mknatsort() - Multi-Key Natural Sort for associative arrays
	*
	*    Uses the uasort() function to perform a natural sort on a multi-dimensional
	*    array on multiple keys. Optionally specifying the sort order for each key
	*    and/or ignoring the case for each key value.
	*
	*    @param array $data_array The array to be sorted.
	*    @param mixed $keys The list of keys to be sorted by. This may be a single
	*        key or an array of keys
	*    @param boolean $reverse Specify whether or not to reverse the sort order. If
	*        there are multiple keys then $reverse may be an array of booleans - one
	*        per key.
	*    @param boolean $ignorecase Specify whether or not to ignore the case when
	*        comparing key values.reverse the sort order. If there are multiple keys
	*        then $ignorecase may be an array of booleans - one per key.
	*/
	function mknatsort ( &$data_array, $keys, $reverse=false, $ignorecase=false ) {
	    // make sure $keys is an array
		if (!is_array($keys)) $keys = array($keys);
		uasort($data_array, sortcompare($keys, $reverse, $ignorecase) );
	}

	function sortcompare($keys, $reverse=false, $ignorecase=false) {
		return function ($a, $b) use ($keys, $reverse, $ignorecase) {
			$cnt=0;
	        // check each key in the order specified
			foreach ( $keys as $key ) {
	            // check the value for ignorecase and do natural compare accordingly
				$ignore = is_array($ignorecase) ? $ignorecase[$cnt] : $ignorecase;
				$result = $ignore ? strnatcasecmp ($a[$key], $b[$key]) : strnatcmp($a[$key], $b[$key]);
	            // check the value for reverse and reverse the sort order accordingly
				$revcmp = is_array($reverse) ? $reverse[$cnt] : $reverse;
				$result = $revcmp ? ($result * -1) : $result;
	            // the first key that results in a non-zero comparison determines
	            // the order of the elements
				if ( $result != 0 ) break;
				$cnt++;
			}
			return $result;
		};
	} // end sortcompare()
