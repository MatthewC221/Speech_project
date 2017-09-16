#!/usr/bin/perl

print "Testing for 1 syllable!";
for (my $i = 0; $i < 6; $i++) {
	my $tmp = "./syl.py words/group1/".$i.".wav 0";
	system($tmp);
	print "\n\n";
}

print "Testing for 2 syllables!"
for (my $i = 0; $i < 6; $i++) {
	my $tmp = "./syl.py words/group2/".$i.".wav 0";
	system($tmp);
	print "\n\n";
}