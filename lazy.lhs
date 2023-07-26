\documentclass[fleqn,leqno,12pt]{article}
\usepackage{varvbtm,amssymb,lh3,natbib,tipa,pstricks,colortab,draftcopy}

%the fix.mhhs.hs script will make the usual literate substitutions on the
%mhhs environment and should be run before compiling with LaTeX

%TODOS:
%	Discuss these in more depth?:
%		a. Make the foot constraint "gradient".
%		b. Tesar: generate epenthesis up to a foot (finite GEN).
%		c. Make the comparison window 3 subsets, not 2.

%syllabification constraints
\newcommand{\ons}{\textsc{Ons}}
\newcommand{\cod}{\textsc{-Cod}}
\newcommand{\prs}{\textsc{Parse}}
\newcommand{\fil}{\textsc{Fill}}
\newcommand{\fb}{\textsc{FtBin}}

%Gen and Eval
\newcommand{\gen}{\textsc{Gen}}
\newcommand{\eval}{\textsc{Eval}}

%numbered verbatim env for code examples
\newverbatim{mhhs}{
\stepcounter{equation}
\bigskip\par
\noindent\parbox{.5in}{(\theequation)}\begin{minipage}[t]{4.7in}
\VVBnonverbmath}
{\small}{}
{\end{minipage}
\par\bigskip}

%equation number; no math; no label
\newcommand{\mheq}[1]{\bigskip\par
\stepcounter{equation}
\ni\parbox{.5in}{(\theequation)}
\begin{minipage}[t]{4.7in} #1 \end{minipage}
\par\bigskip}

%angled brackets
\newcommand{\la}{\ensuremath{\langle}}
\newcommand{\ra}{\ensuremath{\rangle}}

%epenthetic elements
\newcommand{\C}{\ensuremath{\mathbb C}}
\newcommand{\V}{\ensuremath{\mathbb V}}
\newcommand{\T}{\ensuremath{\mathbb T}}
\newcommand{\A}{\ensuremath{\mathbb A}}

%unsyllabified elements
\newcommand{\uc}{\ensuremath{\la\mathrm C\ra}}
\newcommand{\uv}{\ensuremath{\la\mathrm V\ra}}

%syllabification environment
\newcommand{\syl}[1]{\ensuremath{\mathrm{#1}}}

%short no-indenting command
\renewcommand\ni\noindent

%pointy finger
\newfont{\db}{pzdr}
\newcommand{\w}{\mbox{\db +}}

%gray for tableaux
\newcommand\lgr\lightgray

\title{\gen\ with Lazy Evaluation\thanks{Thanks to Amy Fountain and
Diane Ohala for useful discussion. All errors are my own.}}
\author{Michael Hammond \\ U.\ of Arizona}

\begin{document}

\maketitle

\section{Introduction}

Optimality Theory (OT) now exists in multiple flavors, e.g.\ orthodox
\citep{mp,ps}, stochastic \citep{boersma,bh}, harmonic \citep{sl}, etc. In
the orthodox version, the derivation proceeds as follows. There is an input
candidate and an infinite set of possible output candidates. There is a
finite set of constraints that assign violations to the output candidates,
and the candidate that violates the least number of constraints is selected
as the surface form.\footnote{Nothing formally requires that there be a
single winning candidate \citep{idsardi,walma,mylogic}, but we can set this
issue aside.} Orthodox OT calculates the winner in terms of strict ranking:
constraints are strictly ordered and a single violation of a higher-ranked
constraint overpowers any number of violations of a lower-ranked
constraint.

In the schematic example below, there is a finite set of constraints ranked
left to right. There is an infinite set of candidates given on the left
side of each row. Violations are marked with asterisks, and the winning
candidate is marked with the pointing hand.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|}
\cline{2-6}
     & /k\ae t/         & A    & B    & \ldots & C \\
\LCC &                  &      &      & \lgr   & \lgr \\
\cline{2-6}
\w   & [k\super h\ae t] & *    &      &        & ***** \\
\ECC
\LCC &                  &      & \lgr & \lgr   & \lgr \\
\cline{2-6}
     & [hit]            & **!  &      &        & \\
\cline{2-6}
     & [h\ae k]         & **!* &      &        & \\
\cline{2-6}
     & [\v cer]         & **!  &      &        & \\
\ECC
\LCC &                  &      &      & \lgr   & \lgr \\
\cline{2-6}
     & [vznork]         & *    & *!   &        & \\
\cline{2-6}
     & \ldots           & *    & *!*  &        & \\
\ECC
\cline{2-6}
\end{tabular}}

\ni Notice first that all the candidates violate constraint~A; hence only
candidates that violate A as little as possible remain in the candidate
set: [k\super h\ae t], [vznork], and the set denoted by ``\ldots''. All but
the first violate the lower-ranked constraint~B, so the first emerges as
the winning pronunciation. Notice too that the winning candidate violates
constraint~C five times, but these violations are irrelevant as
higher-ranked constraints have already selected [k\super h\ae t] as the
winning pronunciation.

The main problem for any implementation of OT is the infinite candidate set
\citep{ellison,myparsing,tesar}. First, if we view the derivation as
analogous to the construction of a constraint tableau as above, then surely
one step in that process would be to list out the candidates. On the most
obvious interpretation of what this would entail, this step would never
terminate and we would never be able to proceed to determining violations
and selecting a winning candidate.

A second problem is that even if we find a mechanism to get all the
candidates listed, we must still assign violations. Again, the simplest
interpretation of what this entails is writing in the asterisks in the
tableau. Since there are an infinite number of rows, any of which could
violate any constraint, the job of determining violations would never end.

In fact, the problem continues on. Even if we could list all the candidates
and get all the violations marked, we would still have to calculate which
candidate violates the fewest constraints, \emph{modulo} strict ranking or
some alternative. This too poses an infinity problem.

There have been several attempts at dealing with these problems.
\citet{myparsing} and \citet{tesar} propose different finite versions of
\gen. \citet{ellison} proposes to represent the candidate set with a finite
automaton, thus allowing the infinite set to be represented with finite
means. Finally, \citet{karttunenOT} elaborates the finite-state approach,
developing an idea from the first paper.

In this paper, we take another approach. Specifically, we show how the
notion of \emph{lazy evaluation} in a functional programming context can be
used to treat \gen. The basic idea behind lazy evaluation is that we
construct infinite sets which are not immediately evaluated. The
programming context is one where we only evaluate such sets when we need to
and we only evaluate as much of them as we need in the context at hand.
These properties, we will show, allow us an elegant treatment of \gen.

The organization of this paper is as follows. First, we outline how lazy
evaluation works, using the functional programming language Haskell
\citep{jones} as our framework.\footnote{We choose Haskell for two reasons.
First, it is a pure functional language. Second, there are free interpreter
and compiler implementations for all platforms (\texttt{ghc} and
\texttt{hugs}). \citet{hutton} is a nice pedagogical introduction to
Haskell.} Next, we provide an implementation of \gen\ using lazy evaluation
to avoid the infinity problems listed above. We then show how the system
works for the case of containment-based syllabification. Finally, we
consider the issues raised by our implementation for phonological theory.

\section{What is Lazy Evaluation?}

Lazy evaluation is a concept from functional programming. To understand the
former, we must first understand the latter. The basic idea behind
functional programming is that a program is a set of functions and
constants. A number like \lstinline{6} or a string like
\lstinline{"phoneme"} are instances of constants. We can also define terms
that refer to constants; for example, we might define $\pi$ as:

\begin{mhhs}
myPi = 3.14159265358979
\end{mhhs}

\ni or \lstinline{myName} as:

\begin{mhhs}
myName = "Ishmael"
\end{mhhs}

The other component of a functional programming language is the notion of a
function, something that pairs a set of values with one particular value.
Addition is a built-in function: any two numbers are paired with a specific
number. For example, the numbers $4$ and $7$ are uniquely paired with $11$.
We can also write our own functions as follows.

\begin{mhhs}
addTwo   :: Int -> Int
addTwo x =  x + 2
\end{mhhs}

\ni The first line is a specification of the type of elements that this
function pairs, in this case, two integers. The second line above defines
\lstinline{addTwo} as a function that, when applied to some
number---represented as \lstinline{x} here---returns that value plus two.
For example, we could invoke this function by writing the function name
before some constant, e.g.\ \lstinline{addTwo 6}, which would produce the
result \lstinline{8} when interpreted.

This is all there is in a strict functional language. Notice, in
particular, that there are no separate variables, as one would find in
familiar languages like Perl, Java, or C. Constants like \lstinline{myPi}
or \lstinline{myName} are immutable once defined.\footnote{One frequent
misunderstanding of functions is that they \emph{change} some set of things
into another and this would seem to be at odds with the notion of
immutability. It is better, therefore, to think of functions as described
in the text.} 

To understand lazy evaluation in a functional programming context, we also
need to understand how \emph{lists} work.\footnote{There are a host of
other functional data structures with the properties we require, but lists
are one of the very simplest, familiar from languages like Lisp, and very
frequently used.} A list is a sequence of elements terminated by the empty
list. When a list is finite, we can represent it in one of two ways, either
as a sequence of elements enclosed in square brackets, e.g.\
\lstinline{[3,5,7,7]}, or more explicitly as a sequence of elements
concatenated with the list construction operator \lstinline{:}, e.g.\
\lstinline{3:(5:(7:(7:[])))}. Both lists terminate with the empty list
\lstinline{[]}, but this is only overt in the latter notation. Notice too
that the list construction operator takes a list as its right operand and a
single element as its left.

We can manipulate lists in functions. Here is a function that returns the
first element of a list:

\begin{mhhs}
myHead        :: [a] -> a
myHead (x:xs) =  x
\end{mhhs}

\ni The function \lstinline{myHead} returns the first element of a
list.\footnote{Here and following, it is useful to write explicit functions
for some functions that are typically already available in the standard
Haskell prelude (library). This allows us to be maximally explicit about
what our functions do. When we do this, the function we write will begin
with the string \lstinline{my}, e.g.\ \lstinline{myHead},
\lstinline{myTail}, \lstinline{myConcat}, etc.} It does this by
pattern-matching on its argument, requiring that its argument be a list
with two parts---a first element \lstinline{x} and the remaining elements
\lstinline{xs}. The same sort of move can be used to write a function that
returns the remaining elements of a list:

\begin{mhhs}
myTail        :: [a] -> [a]
myTail (x:xs) =  xs
\end{mhhs}

Finally, we can write functions that manipulate lists recursively. Here is
a more complex function that returns the first $n$ elements of a list:

\begin{mhhs}
myTake          :: Int -> [a] -> [a]
myTake 0 _      =  []
myTake n (x:xs) =  x:myTake (n-1) xs
\end{mhhs}

\ni This function takes two arguments, a number and a list. If the number
is $0$, then the function returns the empty list; it doesn't matter what
the second argument is. If the number is greater than $0$, then the
function returns the first element of the list concatenated with the result
of applying the function to the next smaller number and the remainder of
the list.

We can show this schematically in steps with the function application
\lstinline{myTake 2 [4,5,6,7]}.

\mheq{\begin{enumerate}
\renewcommand{\labelenumi}{\alph{enumi}.}

\item\lstinline{myTake 2 [4,5,6,7]}

\item\lstinline{4:(myTake 1 [5,6,7])}

\item\lstinline{4:(5:(myTake 0 [6,7]))}

\item\lstinline{4:(5:[])}

\item\lstinline{[4,5]}

\end{enumerate}}

\ni We begin by applying the function with arguments \lstinline{2} and
\lstinline{[4,5,6,7]}. Remembering that the latter is equivalent to
\lstinline{4:(5:(6:(7:[])))}, the result is the concatenation of
\lstinline{4} with \lstinline{myTake 1 [5,6,7]}. We then evaluate the
embedded \lstinline{myTake} call, producing \lstinline{5:myTake 0 [6,7]}.
Finally, the last call produces \lstinline{[]}, and we assemble all the
bits into \lstinline{[4,5]}.

This mode of interpretation is, in fact, the way Haskell proceeds: from
outside down through embeddings. Lazy evaluation refers to the fact that
evaluation \emph{only occurs when required and only as much as is
required}. In the example above, we go through the list argument only as
far as necessary; it doesn't matter what follows the number 5, since
\lstinline{myTake} is satisfied at that point.

We can see this with an \emph{infinite} list. Here's how we can define a
recursive function that returns an infinite series of numbers:

\begin{mhhs}
infnum   :: Int -> [Int]
infnum x =  x:infnum (x+1)
\end{mhhs}

\ni Consider how this would work when invoked as \lstinline{infnum 1}.

\mheq{\begin{enumerate}
\renewcommand{\labelenumi}{\alph{enumi}.}

\item\lstinline{infnum 1}

\item\lstinline{1:(infnum 2)}

\item\lstinline{1:(2:(infnum 3))}

\item\lstinline{1:(2:(3:(infnum 4)))}

\item\ldots

\end{enumerate}}

\ni Like \lstinline{myTake}, the definition of \lstinline{infnum} is
recursive. However, unlike \lstinline{myTake}, there is no \emph{exit}
clause; there is no mechanism to stop the recursion. Thus, invoking this
command directly will result in the system trying to produce an infinite
series of numbers.

Consider, however, what happens when we invoke \lstinline{infnum 1}
\emph{inside} a call to \lstinline{myTake}:

\mheq{\begin{enumerate}
\renewcommand{\labelenumi}{\alph{enumi}.}

\item\lstinline{myTake 2 (infnum 1)}

\item\lstinline{myTake 2 (1:infnum 2)}

\item\lstinline{1:(myTake 1 (infnum 2))}

\item\lstinline{1:(myTake 1 (2:infnum 3))}

\item\lstinline{1:(2:(myTake 0 (infnum 3))}

\item\lstinline{1:(2:[])}

\item\lstinline{[1,2]}

\end{enumerate}}

\ni We begin with a call to \lstinline{infnum 1} embedded in a call to
\lstinline{myTake} with the argument \lstinline{2}. If its first argument
is greater than $0$, \lstinline{myTake} requires its second argument be
interpretable as a list with a first element: \lstinline{x:xs}. We must
therefore interpret down one more level to see if \lstinline{infnum 1} can
be parsed in this way. Step~b above shows that it can. Step~c shows that we
can then interpret \lstinline{myTake} one step further, altering its first
argument to \lstinline{1}. We repeat the cycle, producing one more call to
\lstinline{myTake}, but this time with the argument \lstinline{0}. Now the
definition of \lstinline{myTake} can return \lstinline{[]} without
interpreting its second argument at all; this is indicated with the single
underscore in the definition of \lstinline{myTake}. The bits are all
assembled and the final result is \lstinline{[1,2]}.

Since this call to \lstinline{myTake} requires no more than two calls to
\lstinline{infnum}, the process terminates. Even though we are manipulating
an object that represents an infinite list, we can do so with impunity
since that object is invoked in a context where it only needs to be
finitely interpreted. This is lazy evaluation.

\section{The Overall Logic}
\label{overall}

The overall logic for our proposal can now be laid out. First, can we
represent the entire candidate set as an infinite list, analogous to
\lstinline{infnum} above? Second, can we represent OT-style constraints as
functions that winnow through such a list, like \lstinline{myTake} above?
We consider each of these problems in turn.

The first problem is superficially straightforward. We can generate an
infinite list of strings quite easily. First, here is code to generate a
list of strings containing only a single symbol. The key new bit here is
that strings are themselves lists; thus a string like \lstinline{"aaa"} is
equivalent to \lstinline{'a':('a':('a':[]))}.

\begin{mhhs}
infa   :: String -> [String]
infa x =  x:infa ('a':x)
\end{mhhs}

The \lstinline{infa} function is actually quite similar to that for
\lstinline{infnum}. The difference is that the new function adds longer and
longer strings to the list, rather than adding larger and larger numbers.
If we invoke this as \lstinline{myTake 5 (infa "")}, then we get:
\lstinline{["","a","aa","aaa","aaaa"]}.

It requires a little more sophistication to get an infinite set of strings
over some alphabet. First we need some utility functions:

\begin{mhhs}
myMap          :: (a -> b) -> [a] -> [b]
myMap f []     =  []
myMap f (x:xs) =  f x:myMap f xs

myPlusplus           :: [a] -> [a] -> [a]
myPlusplus [] ys     =  ys
myPlusplus (x:xs) ys =  x:myPlusplus xs ys

myConcat        :: [[a]] -> [a]
myConcat []     =  []
myConcat (x:xs) =  myPlusplus x (myConcat xs)
\end{mhhs}

\ni The \lstinline{myMap} function takes a function and a list of elements
and applies the function to each element in the list producing a new
list.\footnote{Functional programming languages typically allow functions
to be used directly as arguments to other functions.} For example,
\lstinline{myMap (+2) [1,6,4]} produces \lstinline{[3,8,6]}. The
\lstinline{myPlusplus} generalizes the list construction operator and
allows us to concatenate two lists. Thus \lstinline{myPlusplus [1,2] [7,4]}
produces \lstinline{[1,2,7,4]}. Finally, the \lstinline{myConcat} function
generalizes \lstinline{myPlusplus} to any number of lists. If we invoke
\lstinline{myConcat} like this: \lstinline{myConcat [[1,2],[4,7],[9,2]]},
this produces \lstinline{[1,2,4,7,9,2]}.

Let's now look at the code to generate the infinite set of all possible
strings over the alphabet $\mathrm{\{a,b,c\}}$. First, we define the set of
letters:

\begin{mhhs}
letters :: String
letters =  "abc"
\end{mhhs}

\ni We then define a function that will take a string and return the list
of strings formed by prefixing each letter of the alphabet to the string.
Thus the invocation \lstinline{pfx "x"} produces
\lstinline{["ax","bx","cx"]}.

\begin{mhhs}
pfx   :: String -> [String]
pfx x =  myMap (:x) letters
\end{mhhs}

\ni We then generalize \lstinline{pfx} so that it does the same to lists of
strings:

\begin{mhhs}
pfxall   :: [String] -> [String]
pfxall x =  myConcat (myMap pfx x)
\end{mhhs}

\ni Invoking this on \lstinline{["ax","bx","cx"]} with
\lstinline{pfxall ["ax","bx","cx"]} produces:

\begin{mhhs}
["aax","bax","cax","abx","bbx","cbx","acx","bcx","ccx"]
\end{mhhs}

Finally, we write a recursive function over \lstinline{pfxall} that creates
lists of strings, each one produced by applying \lstinline{pfxall} to the
previous list. The function then joins all the results together with
\lstinline{myPlusplus}.

\begin{mhhs}
infstrings   :: [String] -> [String]
infstrings x =  myPlusplus x (infstrings (pfxall x))
\end{mhhs}

\ni Invoking this directly with \lstinline{infstrings [""]} would produce
an infinite list of strings. We can use lazy evaluation and force only the
first 30 strings to be produced with a call like this:
\lstinline{myTake 30 (infstrings [""])}. This produces the following
output:

\begin{mhhs}
["","a","b","c","aa","ba","ca","ab","bb","cb",
"ac","bc","cc","aaa","baa","caa","aba","bba",
"cba","aca","bca","cca","aab","bab","cab",
"abb","bbb","cbb","acb","bcb"]
\end{mhhs}

Assuming we can encode a phonological representation as a string of
symbols, this establishes that \gen\ can be formalized in terms of lazy
evaluation. Every possible string over the basic segmental vocabulary will
be generated by this set of functions.

The phonological representation is, of course, richer than a simple string
of segments, but this is not a substantive problem. Haskell, like any other
programming language, can accommodate whatever data structures we might
wish to define. As long as we can commit ourselves to some coherent
structural implementation of any bit of nonlinear phonology, we can
implement a lazy \gen\ using that structure.

Let's now turn to the question of how to winnow through such a set.
Basically, we need something to implement \eval\ over the results of
\gen. There are two general strategies. One possibility is to posit a
function that checks every element for some property. This check would be a
function itself that returned the boolean values \lstinline{True} and
\lstinline{False}. We can write this function as below:

\begin{mhhs}
myFilter          :: (a -> Bool) -> [a] -> [a]
myFilter _ []     =  []
myFilter f (x:xs) =  if f x
                     then x:myFilter f xs
                     else myFilter f xs
\end{mhhs}

\ni The \lstinline{myFilter} function applies the function \lstinline{f} to
every element \lstinline{x} in a list, keeping that element if
\lstinline{f x} returns \lstinline{True}. We can use the built-in function
\lstinline{even} to test this with lists of numbers. If we invoke
\lstinline{myFilter} with \lstinline{even} as in:
\lstinline{myFilter even [4,1,2,7]}, we get \lstinline{[4,2]}.

Of course, if we use \lstinline{myFilter} and \lstinline{even} with an
infinite list of numbers, the operation will never terminate. Thus a call
like \lstinline{myFilter even (infnum 1)} goes on forever. We can avoid
this, of course, by embedding this call in an invocation of
\lstinline{myTake}: \lstinline{myTake 10 (myFilter even (infnum 1))}. 
The latter will return \lstinline{[2,4,6,8,10,12,14,16,18,20]}.

Something like \lstinline{myFilter} is fine for constraints that allow for
an infinite number of well-formed candidates. For example, a constraint
like \ons, which penalizes any syllable that has no onset, produces an
infinite set of well-formed candidates. This obtains because the only
mechanism for making the candidate set infinite is epenthesis and \ons\
will let pass all those multiply epenthesized forms where all syllables
have onsets.

There are, however, other constraints that cannot be modeled in this way,
constraints that reduce the infinite candidate set to a finite subset. For
example, a constraint like \fil, which penalizes epenthesis, will rule out
any candidate that has epenthetic elements. The remaining candidate set is
finite.\footnote{This assumes, of course, that the set of nonlinear
structures and possible segments is finite.}

To accommodate constraints like \fil, we need something else. Specifically,
we must assume that the candidates are sorted, such that as we progress
through the set, the number of epenthetic elements increases. Second, we
need a function that tests each element for a property, but that terminates
as soon as that property is not met. Here is how that would look.

\begin{mhhs}
myTakeWhile          :: (a -> Bool) -> [a] -> [a]
myTakeWhile _ []     =  []
myTakeWhile f (x:xs) =  if f x
                        then x:myTakeWhile f xs
                        else []
\end{mhhs}

\ni This function takes elements from the front of a list as long as some
property is satisfied. Once the property does not hold, no additional
elements are taken. The difference from \lstinline{myFilter} is the
\lstinline{else} clause: in the case of \lstinline{myFilter}, we invoke the
function on the remainder of the list; in the case of
\lstinline{myTakeWhile}, we stop processing and return the empty list. If
we invoke the second function on the same list with
\lstinline{myTakeWhile even [4,1,2,7]}, we get \lstinline{[4]} because the
function fails when it reaches the second element of the list.

Consider the different behaviors of these constraints when invoked with a
predicate like \lstinline{(<5)}, which tests for whether its argument is
less than 5. We invoke these as below:

\begin{mhhs}
myTakeWhile (<5) (infnum 1)
myFilter (<5) (infnum 1)
\end{mhhs}

\ni In the first case, the function returns \lstinline{[1,2,3,4]}. In the
second case, the function continues forever. In the case of
\lstinline{myTakeWhile}, the function succeeds with 1 through 4. When it
reaches 5, it terminates because 5 fails the test. In the case of
\lstinline{myFilter}, the function succeeds with 1 through 4, fails with 5,
but continues on looking for numbers less than 5\ldots which, of course, it
will never find.

As functions for winnowing through an infinite list, \lstinline{myFilter}
has the advantage that it will find every element in the string that
matches the test. It has the disadvantage that it will look forever if the
list is infinite. The \lstinline{myTakeWhile} function will only work if
all the cases to be returned are at the beginning of the list. On the other
hand, it has the advantage that it will terminate definitively if the list
is sorted appropriately. We need both sorts of functions.

\section{A Test Case}

To assess whether lazy evaluation along the lines we've sketched here will
work, we will take a simple test case: containment-based syllabification
\citep{ps}.

The basic idea behind containment is that the input and output cannot
differ in terms of the segments they contain, but only in terms of how the
output is syllabified. This restriction on the input-output mapping limits
us to adding syllable structure and manipulating how elements in the input
are parsed or not parsed by that structure. Since there is no bound on the
number of syllables that can be assigned to the output and no restriction
against syllables with vacuous terminal nodes, there are an unbounded
number of candidate output forms. There is no other mechanism deriving the
infiniteness of the candidate set.

Consider, for example a hypothetical input \syl{/CV/}. We represent
syllable boundaries---where necessary---with period (full stop), epenthetic
elements with \C\ or \V, and unparsed segments with angled brackets, e.g.\
\uc\ or \uv. If we exclude epenthetic elements, we have just these four
possibilities: \syl{[CV]}, \syl{[\uc V]}, and [\uc\uv]. Following
\citeauthor{ps}, we only consider syllable parsings that are canonically
well-formed, i.e.\ \syl{[.V.]}, \syl{[.CV.]}, \syl{[.VC.]}, and
\syl{[.CVC.]}, ruling out \syl{[C\uv]}.

If we add in epenthetic elements, the set of possible pronunciations
expands infinitely. Let's start with one epenthetic element. Here is an
exhaustive list of all 11 syllabifications of /CV/ possible with only one
epenthetic segment.

\mheq{\begin{tabular}[t]{c|c|c|c}
\syl{CV}     & \syl{\uc V}    & \syl{(C\uv)}  & \syl{\uc\uv} \\
\hline
\syl{\V.CV}  & \syl{\uc \V.V} & \syl{\V C\uv} & \syl{\V\uc\uv} \\
\syl{\V C.V} & \syl{\uc\C V}  & \syl{C\V\uv} \\
\syl{CV\C}   & \syl{\uc V\C}  & \\
\syl{CV.\V}  & \syl{\uc V.\V} & 
\end{tabular}}

\ni The relative order of epenthetic elements and unparsed elements or
syllable boundaries are not contrastive. Thus \syl{[\uc\V.V]},
\syl{[\V\uc.V]}, and \syl{[\V.\uc V]} are identical. Notice too that even
though \syl{[C\la V\ra]} is not itself a legal candidate, we can generate
legal candidates from it with epenthesis.

We can continue on in like vein. Here is a table of all 38 candidates with
two epenthetic elements.

\mheq{\begin{tabular}[t]{cc|c|c|c}
\multicolumn{2}{c|}{\syl{CV}} & \syl{\uc V} & \syl{(C\uv)} & \syl{\uc\uv} \\
\hline
\syl{CV.\V.\V}    & \syl{\C\V.CV}   & \syl{\C\V\uc.V} &
\syl{\C\V C.\uv}  & \syl{\C\V\uc\uv} \\
\syl{CV.\V\C}     & \syl{\C\V C.V}  & \syl{\V.\C\uc V} &
\syl{\V.\V C\uv}  & \syl{\V\C\uc\uv} \\
\syl{\V.C\V.V}    & \syl{\V.\V.CV}  & \syl{\V.\V\uc.V} &
\syl{\V C.\V\uv}  & \syl{\V.\V\uc\uv} \\
\syl{\V C.\V.V}   & \syl{\V.\V C.V} & \syl{\V\uc.V\C} &
\syl{\V.C\V\uv}   & \\
\syl{\V.CV.\V}    & \syl{\V\C.CV}   & \syl{\C\uc V\C} &
\syl{C\V\C\uv}    & \\
\syl{\V C.V.\V}   & \syl{CV.\C\V}   & \syl{\V\uc.V.\V} &
\syl{C\V.\V\uv}   & \\
\syl{\V C.\C V}   & \syl{CV\C.\V}   & \syl{\C\uc V.\V} & & \\
\syl{\V.CV\C}     & \syl{\V C.V\C}  & \syl{\V\C.\uc V} & & \\
\syl{C\V.\C V}    & \syl{C\V\C.V}   & & & \\
\syl{C\V.V\C}     & \syl{C\V.\V.V}  & & & \\
\syl{C\V.V.\V}    & & & &
\end{tabular}}

We will see that it is essential in our account of \gen\ that the effects
of epenthesis be orderable, though other orderings are possible. Moreover,
we order candidates into bins, each of which is finite in size. Ordering by
epenthesis as above satisfies both requirements.

Another possibility is building an ordering of candidates on the number of
syllables in the candidate. If there are no syllables in the candidate,
there is only one: [\uc\uv]. If there is a single syllable, then we get the
following 14 candidates:

\mheq{\begin{tabular}[t]{c|c|c|c}
\uc\uv       & \syl{CV}   & \syl{\uc V}     & \syl{C\uv} \\
\hline
\uc\uv\V     & \syl{CV}   & \syl{\uc V}     & \syl{C\uv\V} \\
\uc\uv\C\V   & \syl{CV\C} & \syl{\uc\C V}   & \syl{C\uv\V\C} \\
\uc\uv\C\V\C &            & \syl{\uc V\C}   & \syl{\V C\uv} \\
\uc\uv\V\C   &            & \syl{\uc\C V\C} & \syl{\C\V C\uv}
\end{tabular}}

Since, as we noted above, the relative order of unparsed segments and
epenthetic elements is not contrastive, we can exclude unparsed elements
from our graphical representations. This is just a matter of presentation,
however, as one can reconstruct the number of unparsed elements by
comparing candidates with the input. With this assumption, we can convert
the table above to the following:

\mheq{\begin{tabular}[t]{c|c|c|c}
       & \syl{CV}   & \syl{V}      & \syl{C} \\
\hline
\V     & \syl{CV}   & \syl{V}      & \syl{C\V} \\
\C\V   & \syl{CV\C} & \syl{\C V}   & \syl{C\V\C} \\
\C\V\C &            & \syl{V\C}    & \syl{\V C} \\
\V\C   &            & \syl{\C V\C} & \syl{\C\V C}
\end{tabular}}

With two syllables, we get 112 candidates. Again, we leave out unparsed
elements for perspecuity.

\mheq{\begin{tabular}[t]{c|c|c}
\V.\V & \V.V & V.\V \\
\V.\C\V & \V.\C V & \V.C\V \\
\V.CV & V.\C\V & \V.\V\C \\
\V.\V C & \V.V\C & V.\V\C \\
\V.\C\V\C & \V.\C\V C & \V.\C V\C \\
\V.C\V\C & \V.CV\C & V.\C\V\C
\end{tabular}}

\mheq{\begin{tabular}[t]{c|c|c}
\C\V.\V & \C\V.V & \C V.\V \\
C\V.\V & C\V.V & CV.\V \\
\C\V.\C\V & \C\V.\C V & \C\V.C\V \\
\C\V.CV & \C V.\C\V & C\V.\C\V \\
C\V.\C V & CV.\C\V & \C\V.\V\C \\
\C\V.\V C & \C\V.V\C & \C V.\V\C \\
C\V.\V\C & C\V.V\C & CV.\V\C \\
\C\V.\C\V\C & \C\V.\C\V C & \C\V.\C V\C \\
\C\V.C\V\C & \C\V.CV\C & \C V.\C\V\C \\
C\V.\C\V\C & C\V.\C V\C & CV.\C\V\C
\end{tabular}}

\mheq{\begin{tabular}[t]{c|c|c}
\V\C.\V & \V\C.V & \V C.\V \\
\V C.V & V\C.\V & \V\C.\C\V \\
\V\C.\C V & \V\C.C\V & \V\C.CV \\
\V C.\C\V & \V C.\C V & V\C.\C\V \\
\V\C.\V\C & \V\C.\V C & \V\C.V\C \\
\V C.\V\C & \V C.V\C & V\C.\V\C \\
\V\C.\C\V\C & \V\C.\C\V C & \V\C.\C V\C \\
\V\C.C\V\C & \V\C.CV\C & \V C.\C\V\C \\
\V C.\C V\C & V\C.\C\V\C
\end{tabular}}

\mheq{\begin{tabular}[t]{c|c|c}
\C\V\C.\V & \C\V\C.V & \C\V C.\V \\
\C\V C.V & \C V\C.\V & C\V\C.\V \\
C\V\C.V & CV\C.\V & \C\V\C.\C\V \\
\C\V\C.\C V & \C\V\C.C\V & \C\V\C.CV \\
\C\V C.\C\V & \C\V C.\C V & \C V\C.\C\V \\
C\V\C.\C\V & C\V\C.\C V & CV\C.\C\V \\
\C\V\C.\V\C & \C\V\C.\V C & \C\V\C.V\C \\
\C\V C.\V\C & \C\V C.V\C & \C V\C.\V\C \\
C\V\C.\V\C & C\V\C.V\C & CV\C.\V\C \\
\C\V\C.\C\V\C & \C\V\C.\C\V C & \C\V\C.\C V\C \\
\C\V\C.C\V\C & \C\V\C.CV\C & \C\V C.\C\V\C \\
\C\V C.\C V\C & \C V\C.\C\V\C & C\V\C.\C\V\C \\
C\V\C.\C V\C & CV\C.\C\V\C
\end{tabular}}

\ni It will turn out that the latter ordering on syllables is empirically
superior to the former ordering based on epenthesis.

Let's now consider the constraints \citeauthor{ps} use to manipulate these
representations. There are four basic ones:

\mheq{\begin{enumerate}
\renewcommand{\labelenumi}{\alph{enumi}.}

\item \prs: Underlying segments must be parsed into syllable structure. 

\item \fil: Syllable positions must be filled with underlying segments.

\item \ons: A syllable must have an onset.

\item \cod: A syllable must \textbf{not} have a coda.

\end{enumerate}}

\ni The first two constraints are \emph{faithfulness} constraints. They
serve to limit epenthesis and deletion. The last two constraints are
\emph{markedness} constraints and militate for the least marked
syllabification.

Ranking is used to get different effects. If a markedness constraint is
ranked above a faithfulness constraint, then the lowest-ranked faithfulness
constraint will determine whether epenthesis or deletion is used to satisfy
that markedness constraint. If, on the other hand, faithfulness constraints
are ranked above markedness, then inputs are syllabified as best they can
without epenthesis or deletion. Here is an example of the former, where
markedness is ranked high and \fil\ is ranked lowest.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|}
\cline{2-6}
     & /V/  & \ons & \cod & \prs & \fil \\
\LCC &      &      & \lgr & \lgr & \lgr \\
\cline{2-6}
     & V    & *!   &      &      & \\
\ECC
\LCC &      &      &      &      & \lgr \\
\cline{2-6}
     & \uv  &      &      & *!   & \\
\cline{2-6}
\w   & \C V &      &      &      & * \\
\ECC
\cline{2-6}
\end{tabular}}

\ni Here, the markedness constraints are ranked at the top, meaning that
the requirements for an onset and that there not be a coda must be met. The
\fil\ constraint is ranked at the bottom of the hierarchy entailing that
these requirements are met by epenthesis. If, instead, \prs\ were ranked at
the bottom, we'd get deletion instead:

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|}
\cline{2-6}
     & /V/  & \ons & \cod & \fil & \prs \\
\LCC &      &      & \lgr & \lgr & \lgr \\
\cline{2-6}
     & V    & *!   &      &      & \\
\ECC
\LCC &      &      &      &      & \lgr \\
\cline{2-6}
\w   & \uv  &      &      &      & * \\
\cline{2-6}
     & \C V &      &      &  *!  & \\
\ECC
\cline{2-6}
\end{tabular}}

The same logic works in the case of codas with a suitable input. The
following two tableaux show how this works for something like /CVC/. First,
we see that epenthesis results when \cod\ is ranked above a faithfulness
constraint and \fil\ is ranked bottommost.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|}
\cline{2-6}
     & /CVC/ & \ons & \cod & \prs & \fil \\
\LCC &       &      &      & \lgr & \lgr \\
\cline{2-6}
     & CVC   &      & *!   &      & \\
\ECC
\LCC &       &      &      &      & \lgr \\
\cline{2-6}
     & CV\uc &      &      & *!   & \\
\cline{2-6}
\w   & CVC\V &      &      &      & * \\
\ECC
\cline{2-6}
\end{tabular}}

\ni Then we see that deletion results when \prs\ is at the bottom.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|}
\cline{2-6}
     & /CVC/ & \ons & \cod & \fil & \prs \\
\LCC &       &      &      & \lgr & \lgr \\
\cline{2-6}
     & CVC   &      & *!   &      & \\
\ECC
\LCC &       &      &      &      & \lgr \\
\cline{2-6}
\w   & CV\uc &      &      &      & * \\
\cline{2-6}
     & CVC\V &      &      & *!   & \\
\ECC
\cline{2-6}
\end{tabular}}

Finally, we see that when both faithfulness constraints are ranked above
the markedness constraints, markedness violations in the input surface as
is.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|}
\cline{2-6}
     & /CVC/ & \fil & \prs & \ons & \cod \\
\LCC &       &      &      & \lgr & \lgr \\
\cline{2-6}
\w   & CVC   &      &      &      & * \\
\cline{2-6}
     & CV\uc &      & *!   &      & \\
\ECC
\LCC &       &      & \lgr & \lgr & \lgr \\
\cline{2-6}
     & CVC\V & *!   &      &      & \\
\ECC
\cline{2-6}
\end{tabular}}

This system provides an account of the fact that, while onsets can be
required, codas can never be, and that while codas can be disallowed,
onsets never are. In fact, \citeauthor{ps} present this as a theorem.

\mheq{Universally Optimal Syllables \\
No language may prohibit the syllable .CV. Thus, no language prohibits
onsets or requires codas.}

The constraints presented have different effects in terms of the finiteness
of the candidate set that they might permit. The constraints \prs, \ons,
and \cod\ permit an infinite candidate set. To see this, note that for each
case, there are an infinite number of possible candidates for /\syl{CV}/
that do not violate the constraint at all.

For \prs, we can generate an infinite number of candidates that are
well-formed by suffixing any number of instances of \V, e.g.\ [\syl{CV}],
[\syl{CV.\V}], [\syl{CV.\V.\V}], [\syl{CV.\V.\V.\V}]. etc. The same set
suffices for \cod. For \ons, we generate an infinite set of well-formed
candidates by appending the sequence \C\V: [\syl{CV}], [\syl{CV.\C\V}],
[\syl{CV.\C\V.\C\V}], [\syl{CV.\C\V.\C\V.\C\V}], etc.

This is impossible for \fil. The only way to generate an infinite candidate
set is with epenthesis, but epenthesis gives rise to violations of \fil.
Hence, the set of candidates that are well-formed with respect to \fil\ is
finite.

We must therefore distinguish between \fil\ violations and the other cases
in terms of technology analogous to the difference between
\lstinline{myTakeWhile} and \lstinline{myFilter}. To do this, we generate
candidates lazily, binning by the number of syllables: $\{B_0, B_1, B_2,
\ldots\}$. Starting at the first bin, we evaluate candidates as usual,
selecting a winner---or winners---for that bin, call this $w(B_0)$. We then
go on to the next bin and evaluate the candidates there, determining the
winner for that bin $w(B_1)$. In the general case, if the winner for some
bin $w(B_n)$ is better than $w(B_{n-1})$, we continue on to $B_{n+1}$. If
$w(B_n)$ is not better than $w(B_{n-1})$, then we are done and the winning
candidate for the entire set is $w(B_{n-1})$.

We can express this algorithm in (procedural) pseudocode as follows:

\mheq{\begin{enumerate}
\renewcommand{\labelenumi}{\alph{enumi}.}

\item Set global winner to null.

\item Go to first bin.

\item Assess violations for all candidates in current bin.

\item Choose winner from current bin.

\item If current winner is better than global winner:

	\begin{enumerate}
	\renewcommand{\labelenumii}{\roman{enumii}.}

	\item set global winner to current winner, and

	\item go to next bin, and

	\item go to (c)

	\end{enumerate}

else end: global winner is winner.

\end{enumerate}}

Let's now look at an example. Consider the candidate /\syl{VC}/ with the
constraint ranking \ons\ $\gg$ \cod\ $\gg$ \prs\ $\gg$ \fil.

The first bin $B_0$ has no syllables; hence all segments are unparsed and
there is only one candidate: [\uv\uc], and it is, of course, the winner,
e.g.\ $w(B_0) = \uv\uc$. We now compare that winner to the winner of $B_1$.
First, we determine $w(B_1)$ as in the following tableau. There are a
finite number of candidates, but for convenience not all candidates are
given.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|}
\cline{2-6}
     & /VC/       & \ons & \cod & \prs & \fil \\
\LCC &            &      & \lgr & \lgr & \lgr \\
\cline{2-6}
     & V\uc       & *!   &      & *    & \\
\cline{2-6}
     & VC         & *!   & *    &      & \\
\ECC
\LCC &            &      &      &      & \lgr \\
\cline{2-6}
     & \C\V\uv\uc &      &      & **!  & ** \\
\cline{2-6}
\w   & \uv C\V    &      &      & *    & * \\
\ECC
\cline{2-6}
\end{tabular}}

We see that $w(B_1) = \syl{\uv C\V}$. We must now compare $w(B_0)$ with
$w(B_1)$. This is shown in the following tableau.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|}
\cline{2-6}
     & /VC/       & \ons & \cod & \prs & \fil \\
\LCC &            &      &      &      & \lgr \\
\cline{2-6}
     & \uv\uc     &      &      & **!  & \\
\cline{2-6}
\w   & \uv C\V    &      &      & *    & * \\
\ECC
\cline{2-6}
\end{tabular}}

Since $w(B_1)$ wins, we must go on to evaluate the candidates of $B_2$.
Again, there are a finite number of candidates, but too many to display
easily, so the following tableau just contains a few of them.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|}
\cline{2-6}
     & /VC/        & \ons & \cod & \prs & \fil \\
\LCC &             &      & \lgr & \lgr & \lgr \\
\cline{2-6}
     & VC\V        & *!   &      &      & * \\
\ECC
\LCC &             &      &      & \lgr & \lgr \\
\cline{2-6}
     & \C VC\V\C   &      & *!   &      & *** \\
\ECC
\LCC &             &      &      &      & \lgr \\
\cline{2-6}
     & \uv C\V\C\V &      &      & *!   & *** \\
\cline{2-6}
\w   & \C VC\V     &      &      &      & ** \\
\ECC
\cline{2-6}
\end{tabular}}

Again, the winner(s) here must be compared with the previous best
candidate(s).

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|}
\cline{2-6}
     & /VC/    & \ons & \cod & \prs & \fil \\
\LCC &         &      &      &      & \lgr \\
\cline{2-6}
     & \uv C\V &      &      & *!   & * \\
\cline{2-6}
\w   & \C VC\V &      &      &      & ** \\
\ECC
\cline{2-6}
\end{tabular}}

Since $w(B_2)$ wins, we must go on to consider $B_3$. Once again, only
representative candidates are given (though the full number is, of course,
finite).

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|}
\cline{2-6}
     & /VC/        & \ons & \cod & \prs & \fil \\
\LCC &             &      &      & \lgr & \lgr \\
\cline{2-6}
\w   & \C\V\C VC\V &      &      &      & **** \\
\cline{2-6}
\w   & \C VC\V\C\V &      &      &      & **** \\
\cline{2-6}
     & \C VC\V\C   &      & *!   &      & *** \\
\ECC
\cline{2-6}
\end{tabular}}

Here two candidates tie for $w(B_3)$, so both must be compared with the
previous winner.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|}
\cline{2-6}
     & /VC/        & \ons & \cod & \prs & \fil \\
\cline{2-6}
\w   & \C VC\V     &      &      &      & ** \\
\cline{2-6}
     & \C\V\C VC\V &      &      &      & ***!* \\
\cline{2-6}
     & \C VC\V\C\V &      &      &      & ***!* \\
\cline{2-6}
\end{tabular}}

Here $w(B_2)$ wins out over the candidates from $w(B_3)$ and \eval\
terminates with [\syl{\C VC\V}] as the overall winning candidate.

The general procedure involves considering the candidate set in increments
determined by syllable structure. At each stage only a finite number of
candidates need be considered and the procedure only goes on to the next
stage if the last stage produces the best candidate up to that point.

Other binning logic would not fare as well. Consider again binning by the
number of instances of epenthesis: $B_0$ would have no epenthesis, $B_1$
only one instance, and so on. The problem here is that sometimes a single
instance of epenthesis will worsen a candidate and only a second instance
of epenthesis will improve it.

Axininca Campa \citep{spring,mp} provides a concrete example. Axininca
stems must satisfy a prosodic minimum in certain contexts. This prosodic
minimum must occasionally be achieved by multiple instances of epenthesis.
For example, the root \emph{na} `carry on shoulder' is realized as
[na$\mathbb{TA}$] with the suffix string [piro$\mathbb
T$aanc\textipa{\super h}i].\footnote{In our discussion of Axininca, we will
follow \citeauthor{mp} in representing epenthetic elements as $\mathbb{T}$
and $\mathbb{A}$, rather than as \C\ and \V.}

The precise constraints that force this are not the issue. Basically, words
must be at least feet and feet must be at least binary. On such an
analysis, a single instance of epenthesis provides no improvement. More
concretely, let's assume an analysis with the following constraints:

\mheq{\cod\ $\gg$ \ons\ $\gg$ \fb\ $\gg$ \prs\ $\gg$ \fil}

\ni The \fb\ constraint forces feet to be binary.

Let's first consider how syllabic bins get the correct result. At $B_0$, we
have only one candidate $w(B_0) = \syl{\la n\ra\la a\ra}$. At $B_1$, we
have:

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|c|}
\cline{2-7}
     & /na/         & \cod & \ons & \fb  & \prs & \fil \\
\LCC &              &      & \lgr & \lgr & \lgr & \lgr \\
\cline{2-7}
\w   & na           &      &      & *    &      & \\
\cline{2-7}
     & na\T         & *!   &      & *    &      & * \\
\cline{2-7}
     & \la n\ra a\T & *!   & *    & *    & *    & * \\
\ECC
\cline{2-7}
\end{tabular}}

We then compare $w(B_0)$ with $w(B_1)$.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|c|}
\cline{2-7}
     & /na/             & \cod & \ons & \fb  & \prs & \fil \\
\LCC &                  &      &      &      &      & \lgr \\
\cline{2-7}
\w   & na               &      &      & *    &      & \\
\cline{2-7}
     & \la n\ra\la a\ra &      &      & *    & *!*  & \\
\ECC
\cline{2-7}
\end{tabular}}

Since $w(B_1)$ is better than $w(B_0)$, we go on to $B_2$.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|c|}
\cline{2-7}
     & /na/    & \cod & \ons & \fb  & \prs & \fil \\
\LCC &         &      &      & \lgr & \lgr & \lgr \\
\cline{2-7}
\w   & na\T\A  &      &      &      &      & ** \\
\ECC
\LCC &         &      & \lgr & \lgr & \lgr & \lgr \\
\cline{2-7}
     & \A na\T & *!   & *    &      &      & ** \\
\ECC
\LCC &         &      &      & \lgr & \lgr & \lgr \\
\cline{2-7}
     & na\A    &      & *!   &      &      & * \\
\ECC
\cline{2-7}
\end{tabular}}

We must now compare $w(B_1)$ and $w(B_2)$.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|c|}
\cline{2-7}
     & /na/   & \cod & \ons & \fb & \prs & \fil \\
\LCC &        &      &      &     & \lgr & \lgr \\
\cline{2-7}
\w   & na\T\A &      &      &     &      & ** \\
\cline{2-7}
     & na     &      &      & *!  &      & \\
\ECC
\cline{2-7}
\end{tabular}}

Since $w(B_2)$ wins, we must go on to determine $w(B_3)$.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|c|}
\cline{2-7}
     & /na/                         & \cod & \ons & \fb  & \prs & \fil \\
\LCC &                              &      &      &      &      & \lgr \\
\cline{2-7}
\w   & na\T\A\T\A                   &      &      &      &      & **** \\
\ECC
\LCC &                              &      &      & \lgr & \lgr & \lgr \\
\cline{2-7}
     & \la n\ra a\T\A\T\A           &      & *!   &      & *    & **** \\
\ECC
\LCC &                              &      &      &      &      & \lgr \\
\cline{2-7}
     & \la n\ra\la a\ra\T\A\T\A\T\A &      &      &      & *!*  & ****** \\
\ECC
\cline{2-7}
\end{tabular}}

Finally, we compare $w(B_2)$ with $w(B_3)$:

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|c|}
\cline{2-7}
     & /na/       & \cod & \ons & \fb  & \prs & \fil \\
\cline{2-7}
\w   & na\T\A     &      &      &      &      & ** \\
\cline{2-7}
     & na\T\A\T\A &      &      &      &      & ***!* \\
\cline{2-7}
\end{tabular}}

\ni Since $w(B_2)$ wins, we are done and have gotten the desired result.

If we were to bin by number of instances of epenthesis, we would not get
the correct result. Let's go through the same derivation with bins by
epenthesis to see this. First, we consider $B_0$. Notice that the number of
syllables is not controlled in this bin, only the number of instances of
epenthesis.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|c|}
\cline{2-7}
     & /na/             & \cod & \ons & \fb  & \prs & \fil \\
\LCC &                  &      &      &      &      & \lgr \\
\cline{2-7}
\w   & na               &      &      & *    &      & \\
\ECC
\LCC &                  &      &      & \lgr & \lgr & \lgr \\
\cline{2-7}
     & \la n\ra a       &      & *!   & *    & *    & \\
\ECC
\LCC &                  &      &      &      &      & \lgr \\
\cline{2-7}
     & \la n\ra\la a\ra &      &      & *    & *!*  & \\
\ECC
\cline{2-7}
\end{tabular}}

We now go on to $B_1$, where every candidate has a single instance of
epenthesis.

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|c|}
\cline{2-7}
     & /na/         & \cod & \ons & \fb  & \prs & \fil \\
\LCC &              &      & \lgr & \lgr & \lgr & \lgr \\
\cline{2-7}
     & na\T         & *!   &      & *    &      & * \\
\ECC
\LCC &              &      &      & \lgr & \lgr & \lgr \\
\cline{2-7}
     & \A na        &      & *!   & *    & *    & * \\
\cline{2-7}
\w   & \T\la n\ra a &      &      & *    & *    & * \\
\ECC
\cline{2-7}
\end{tabular}}

We now compare $w(B_0)$ with $w(B_1)$:

\mheq{\begin{tabular}[t]{r|c|c|c|c|c|c|}
\cline{2-7}
     & /na/         & \cod & \ons & \fb  & \prs & \fil \\
\LCC &              &      &      &      &      & \lgr \\
\cline{2-7}
\w   & na           &      &      & *    &      & \\
\cline{2-7}
     & \T\la n\ra a &      &      & *    & *!   & * \\
\ECC
\cline{2-7}
\end{tabular}}

\ni Here, $w(B_0)$ wins, so the algorithm terminates with the incorrect
result. The problem is that we have found a local minimum in optimality
before we reached $B_2$. Hence, if we have captured the essential
properties of the Axininca analysis correctly, binning by the number of
instances of epenthesis is empirically inadequate.

On the other hand, we have seen that for containment-based OT,
syllable-based binning works with lazy \gen.

\section{Remaining issues}

There are several remaining issues.

One issue is whether the model generalizes to correspondence-based
Optimality Theory \citep{ct}. In this version of OT, anything can change
from input to output and we are not bound to containment.

Correspondence-based OT is not a problem for lazy \gen. We've already seen
in Section~\ref{overall} that we can generate an infinite candidate set
over any finite alphabet. If we syllabify those candidates, we can just as
easily bin them by the number of syllables each candidate contains. If, on
the other hand, the alphabet is not finite, then there would indeed be a
problem.

A second general question concerns the nature of the bins. Syllable-based
bins will not generalize to other domains, e.g.\ morphology or syntax. We
are not committed to syllable-based bins for every possible domain of
grammar. It may very well be that bins in other domains are empirically
determined. We are committed to the position that some binning will work in
all other domains, because lazy \gen\ requires bins.

A third remaining issue is whether syllable-based bins are adequate for
phonological theory. The prediction of syllable-based bins is that the best
candidate will never be more than a bin further along than the previous
best candidate. In more formal terms, we cannot have a situation whether
$w(B_n)$ is the true winner, but $w(B_{n-1})$ loses to $w(B_{n-2})$.

What would such a case look like? Imagine a language like Axininca, but
where the optimal candidate must be at least \emph{three} syllables long,
i.e. [na\T\A\T\A]. Syllable-based bins with lazy \gen\ would not find this
candidate.

\bibliographystyle{linquiry2}
\bibliography{../allbib}

\appendix

\section{Notes on the Implementation}

The proposal outlined in the text is implemented in Haskell here as a
demonstration proof that the system works.

This paper is written in \emph{literate Haskell} style, which means that
the code and the paper derive from the same source code. This file thus
constitutes the working code and the paper that describes it.

For convenience, unparsed elements are not indicated in output forms. Thus
one candidate output for input /\syl{hat}/ is [\syl{ha\C}], with an
unparsed [\syl{t}] and an epenthetic [\C]. This would be equivalent to
[\syl{ha\la t\ra\C}] or [\syl{ha\C\la t\ra}]. It is possible to reconstruct
the number of unparsed elements in an output by comparing it with the input
and this is how the implementation of \prs\ works below.

The program can be invoked from within the \texttt{ghci} or \texttt{hugs}
interpreters by calling the function \lstinline{eval} with two arguments.
The first argument is the input form in double quotes. The second argument
is an ordered list of constraints in square brackets, separated by commas.
For example:

\begin{mhhs}
eval "hat" [onset,nocoda,fill,parse]
\end{mhhs}

\ni Alternatively, the program can be compiled with the \texttt{ghc}
compiler and run on the command-line like this:

\begin{mhhs}
./lazy hat onset nocoda fill parse
\end{mhhs}

\ni Lastly, the program can be run in a one-off mode with
\texttt{runhaskell} like this:

\begin{mhhs}
runhaskell lazy.lhs hat onset nocoda fill parse
\end{mhhs}

\section{Implementation}

\begin{code}
import List (isPrefixOf)
import System.Environment (getArgs)

--constraints make a number from an input and and output
type Constraint = String -> String -> Int

--a candidate is a string and a vector of violations
type Candidate  = (String,[Int])

--to run the program on its own
main = do as <- getArgs
          if (length as) < 2
            then error "usage: lazy input c1 c2 c3..."
            else do let i = head as
                    let cs = map convert $ tail as
                    putStr $ unlines $ map fst $ eval i cs

--converts strings to constraints
convert          :: String -> Constraint
convert "onset"  =  onset
convert "nocoda" =  nocoda
convert "fill"   =  fill
convert "parse"  =  parse
convert "ftbin"  =  ftbin
convert x        =  error (x ++ " is not a constraint name")

--entry function for eval
eval      :: String -> [Constraint] -> [Candidate]
eval i cs =  evl [] $ map (makeCanVecs i cs) (gen i)

--evaluates bin by bin, called by eval
evl               :: [(String, [Int])] -> [[(String, [Int])]]
                     -> [(String, [Int])]
evl [] (y:ys)     =  evl (getBest [] y) ys
evl (x:xs) (y:ys) =  if rank (>) (head best) x
                       then (x:xs)
                       else if rank (==) (head best) x
                              then evl ((x:xs) ++ best) ys
                              else evl best ys
                       where best = getBest [] y

--gets the highest-ranked candidates from a set
getBest            :: [Candidate] -> [Candidate] -> [Candidate]
getBest xs []      =  xs
getBest [] (y:ys)  =  getBest [y] ys
getBest (x:xs) (y:ys)
   | rank (==) x y =  getBest (y:x:xs) ys
   | rank (<) x y  =  getBest (x:xs) ys
   | otherwise     =  getBest [y] ys

--compares the ranking of two candidate,vector pairs
rank       :: ([Int] -> [Int] -> Bool) -> Candidate ->
              Candidate -> Bool
rank c a b =  c (snd a) (snd b)

--makes a set of candidate,vector pairs for a bin
makeCanVecs             :: String -> [Constraint] -> [String] ->
                           [Candidate]
makeCanVecs _ _ []      =  []
makeCanVecs i xs (c:cs) =  (c,makeVec i xs c):makeCanVecs i xs cs

--makes a vector of violations for a ranked set of constraints
makeVec            :: String -> [Constraint] -> String -> [Int]
makeVec _ [] _     =  []
makeVec i (x:xs) c =  x i c:makeVec i xs c

--NOCODA constraint
nocoda      :: Constraint
nocoda _ "" =  0
nocoda i c  =  if length c > 1 && c!!1 == '.' && isConsonant (c!!0)
                 then 1 + (nocoda i (tail c))
                 else nocoda i (tail c)

--ONSET constraint
onset      :: Constraint
onset _ "" =  0
onset i c  =  if length c > 1 && c!!0 == '.' && isVowel (c!!1)
                then 1 + (onset i (tail c))
                else onset i (tail c)

--PARSE constraint
parse     :: Constraint
parse i c =  (length i) - (((length c) - (fill i c)) -
             (count "." c))

--FILL constraint
fill     :: Constraint
fill _ c =  (count "V" c) + (count "C" c)

--FTBIN constraint
ftbin     :: Constraint
ftbin _ c =  if (count "." c) > 2 then 0 else 1

--counts how many times something occurs in a string
count      :: String -> String -> Int
count _ "" =  0
count p s  =  if isPrefixOf p s then 1 + (count p (tail s))
                                else count p (tail s)

--creates the infinite candidate set
gen   :: String -> [[String]]
gen w =  gn w 0 where gn w n = makeBin w n:gn w (n+1)

--makes a single syllable bin
makeBin     :: String -> Int -> [String]
makeBin w n =  concat $ map (makeSyl w) (polysyllables n)

--makes all parses of an input for a single template
makeSyl      :: String -> String -> [String]
makeSyl _ "" =  [""]
makeSyl w s  =  map fst $ doAllSubs ((length s)-1) [(s,w)]

--does n substitutions in a list of templates+inputs
doAllSubs      :: Int -> [(String,String)] -> [(String,String)]
doAllSubs 0 ps =  concat $ map (doSubs 0) ps
doAllSubs n ps =  concat $ map (doSubs n) (doAllSubs (n-1) ps)

--makes all substitutions for a particular position in template
doSubs          :: Int -> (String,String) -> [(String,String)]
doSubs n (ps,w) =  (ps,w):map makePairs fixedBits
                   where
                   makePairs x = (makeSub (fst x) ps n,snd x)
                   fixedBits = filter
                                 (segType (ps!!n) . fst)
                                 theBits
                   theBits = map (bits w) [0..(length w)-1]

--substitutes an indexed character in a string
makeSub       :: Char -> String -> Int -> String
makeSub c s n =  (take n s) ++ [c] ++ (drop (n+1) s)

--gets segment type by C,V
segType       :: Char -> Char -> Bool
segType 'C' x =  isConsonant x
segType 'V' x =  isVowel x
segType '.' _ =  False

--returns the nth character plus remainder of the string
bits        :: String -> Int -> (Char,String)
bits w n =  (w!!n,drop (n+1) w)

--tests for consonanthood
isConsonant :: Char -> Bool
isConsonant =  not . isVowel

--tests for vowelhood
isVowel   :: Char -> Bool
isVowel v =  elem v vowels

--set of recognized vowels
vowels :: String
vowels =  "Vaeiou"

--set of possible syllables
syllables :: [String]
syllables =  words "V CV VC CVC"

--generates the set of polysyllabic shapes
polysyllables   :: Int -> [String]
polysyllables 0 =  [""]
polysyllables n =  map (++".") (polysyls n)

--called by polysyllables to make the shapes
polysyls   :: Int -> [String]
polysyls 0 =  [""]
polysyls n =  concat (map sylPfx (polysyls (n-1)))

--prefixes all syllable types to a shape
sylPfx   :: String -> [String]
sylPfx x =  map ((x++".")++) syllables
\end{code}

\end{document}

