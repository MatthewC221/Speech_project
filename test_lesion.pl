#!/usr/bin/perl

@correct_words = ("li", "sion");

for (my $tmp = 0; $tmp < 10; $tmp++) {
	$syl_str = "./syl.py words/lesion/lesion".$tmp.".wav 0";
	system($syl_str);
	$total = $#correct_words + 1;
	for (my $i = 0; $i < $total; $i++) {
		my $string = "./mfcc.py f".$i." 0 -1 ".$correct_words[$i];
		system($string);
		print "\n";
	}
	`rm *.wav`;
}