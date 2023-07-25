use LWP::Simple;

if ($#ARGV != 2) {
	die("usage: perl websearch.pl URL pattern hits\n");
}

$linkMax = 50;
push(@urlstodo, $ARGV[0]);
$pattern = $ARGV[1];
$hitmax = $ARGV[2];

while ($hits < $hitmax and $#urlstodo > -1) {
	$linksChecked++;
	if ($linksChecked % 10 == 0) {
		print(STDERR "Checked: $linksChecked\n");
	}
	$theURL = pop(@urlstodo);
	push(@done, $theURL);
	$mycontent = get($theURL);
	if ($mycontent) {
		@lines = split(/\n/, $mycontent);
		foreach $line (@lines) {
			if ($line =~ /$pattern/) {
				print("$theURL:\t$line\n\n");
				$hits++;
			}
			if ($line =~ s/^.*(href|HREF)="([^"]+\.html?)".*$/\2/) {
				addURL($theURL, $line);
			}
		}
	}
}

#adds the url to the list of urls if it's not already there and hasn't already
#been checked.
sub addURL {
	my($url);
	if ($#urlstodo < $linkMax) {
		my($prefix) = shift();
		my($suffix) = shift();
		if ($suffix !~ /^http/) {
			if ($prefix =~ /\/$/) {
				$suffix = $prefix . $suffix;
			} else {
				$suffix = $prefix . "/" . $suffix;
			}
		}
		foreach $url (@done) {
			if ($suffix eq $url) {
				return();
			}
		}
		foreach $url (@urlstodo) {
			if ($suffix eq $url) {
				return();
			}
		}
		push(@urlstodo, $suffix);
	}
}

