#!/usr/bin/perl

use strict;
use warnings;



my $vocab = "hw2_data/spam/vocab.dat";
my %vocab;
&vocabPreprocess($vocab);


my $ionosphere_train = "hw2_data/ionosphere/ionosphere_train.dat";
my $ionosphere_test = "hw2_data/ionosphere/ionosphere_test.dat";
&ionospherePreprocess($ionosphere_train);
&ionospherePreprocess($ionosphere_test);

my $email_train_spam = "hw2_data/spam/train/spam/train_spam";
my $email_train_ham = "hw2_data/spam/train/ham/train_ham";
my $email_test_spam = "hw2_data/spam/test/spam/test_spam";
my $email_test_ham = "hw2_data/spam/test/ham/test_ham";
&emailPreprocess($email_train_spam);
&emailPreprocess($email_train_ham);
&emailPreprocess($email_test_spam);
&emailPreprocess($email_test_ham);


sub ionospherePreprocess {
    my $file = shift;
    my $outfile = $file . ".final";
    open(FILE, "$file") or die "cannot open file $file\n";
    open(OUT, ">$outfile") or die "cannot open file $outfile\n";
    while(<FILE>){
        chomp;
        my @line = split(",", $_);
        if($line[-1] eq "b"){
            $line[-1] = 1;
        }elsif($line[-1] eq "g"){
            $line[-1] = 0;
        }
        my $printLine = join(" ", @line);
        print OUT "$printLine\n";
    }
}


sub emailPreprocess {
    my $file = shift;
    my $outfile = $file . ".final";
    open(LI, "$file") or die "cannot open file $file\n";
    open(OUT, ">$outfile") or die "cannot open file $outfile\n";
    my @sortkeysVocab = sort keys %vocab;
    my $printSortkeys = join(" ", @sortkeysVocab);
    my $lastelementIndex = $#sortkeysVocab;
    print OUT "$printSortkeys\n";
    while(<LI>){
        chomp;
        my %localVocab;
        open(FILE, "$_") or die "cannot open file $_\n";
        while(<FILE>){
            chomp;
            my $line = lc($_);
            my @line = split(/[\.?,\s]+/, $line);
            for(0..$#line){
                $localVocab{$line[$_]} ++;
            }
        };
        my $i = 0;
        foreach my $key (sort keys %vocab){
            $localVocab{$key} = 0 unless exists $localVocab{$key};
            if($i < $lastelementIndex){
                print OUT "$localVocab{$key} ";
            }else{
                print OUT "$localVocab{$key}\n";
            }
            $i ++;
        };

    }
}


sub vocabPreprocess{
    my $file = shift;
    open(FILE, "$file") or die "cannot open file $file\n";
    while(<FILE>){
        chomp;
        $vocab{$_} = 1;
    }
}






