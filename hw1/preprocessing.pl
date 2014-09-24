#!/usr/bin/perl

use strict;
use warnings;

open(IN, "$ARGV[0]") or die "cannot open file $ARGV[0]\n";
my $out = "$ARGV[0].out.x";
my $outy = "$ARGV[0].out.y";
open(OUT, ">$out") or die "cannot open file $out\n";
open(OUTY, ">$outy") or die "cannot open file $outy\n";

while(<IN>){
    chomp;
    my @line = split(",", $_);
    
    my ($outputstringb, $outputstringm, $outputstringd, $outputstringp, $outputstringl, $outputstrings, $outputstringy);
    
    if($line[0] eq "vhigh"){
        $outputstringb = "1\t0\t0\t0";
    }elsif($line[0] eq "high"){
        $outputstringb = "0\t1\t0\t0";
    }elsif($line[0] eq "med"){
        $outputstringb = "0\t0\t1\t0";
    }elsif($line[0] eq "low"){
        $outputstringb = "0\t0\t0\t1";
    };
    
    if($line[1] eq "vhigh"){
        $outputstringm = "1\t0\t0\t0";
    }elsif($line[1] eq "high"){
        $outputstringm = "0\t1\t0\t0";
    }elsif($line[1] eq "med"){
        $outputstringm = "0\t0\t1\t0";
    }elsif($line[1] eq "low"){
        $outputstringm = "0\t0\t0\t1";
    };
    
    if($line[2] eq "2"){
        $outputstringd = "1\t0\t0\t0";
    }elsif($line[2] eq "3"){
        $outputstringd = "0\t1\t0\t0";
    }elsif($line[2] eq "4"){
        $outputstringd = "0\t0\t1\t0";
    }elsif($line[2] eq "5more"){
        $outputstringd = "0\t0\t0\t1";
    };
    
    if($line[3] eq "2"){
        $outputstringp = "1\t0\t0";
    }elsif($line[3] eq "4"){
        $outputstringp = "0\t1\t0";
    }elsif($line[3] eq "more"){
        $outputstringp = "0\t0\t1";
    };
    
    
    if($line[4] eq "small"){
        $outputstringl = "1\t0\t0";
    }elsif($line[4] eq "med"){
        $outputstringl = "0\t1\t0";
    }elsif($line[4] eq "big"){
        $outputstringl = "0\t0\t1";
    };
    
    if($line[5] eq "low"){
        $outputstrings = "1\t0\t0";
    }elsif($line[5] eq "med"){
        $outputstrings = "0\t1\t0";
    }elsif($line[5] eq "high"){
        $outputstrings = "0\t0\t1";
    };
    
    if($line[6] eq "unacc"){
        $outputstringy = 1;
    }elsif($line[6] eq "acc"){
        $outputstringy = 2;
    }elsif($line[6] eq "good"){
        $outputstringy = 3;
    }elsif($line[6] eq "vgood"){
        $outputstringy = 4;
    }
    
    
    print OUT "$outputstringb\t$outputstringm\t$outputstringd\t$outputstringp\t$outputstringl\t$outputstrings\n";
    print OUTY "$outputstringy\n";
}
