# comBiTeS
<h1>Compacting binary and ternary spaced seeds</h1>

Software tools to work with genetic sequences (reference sequences and short reads) and exploit SIMD instructions for better performance. Spaces seeds are masks that allow us to account for or ignore pointwise differences between two sequences. We can use the seeds to create a library of records (signature, position) for a reference sequence, thus helping us find possible alignments of a read to the reference sequence. Signatures are formed based on non-zero symbols for a spaced seed and are effectively some arrays of bits. As zero elements of seeds provide us with the same bits (independent of actual nucleotide values), we want to ignore them to save space when storing the library. Note that we may shuffle the resulting bits for a signature. While we produce another array of bits, this does not affect the sequence alignment process. Therefore, when calculating signature values, we may shuffle symbols to fill all gaps created by zero seed elements.

<nav>
  <ul>
    <li><a href="#link_storage">Storage</a></li>
    <li><a href="#link_seed">Spaced seeds</a></li>
    <li><a href="#link_ternary">Ternary spaced seeds</a></li>	  
    <li><a href="#link_form">Forming signatures</a></li>
    <li><a href="#link_tools">Tools</a></li>
  </ul>
  </nav>

<h2 id="link_storage">Storage</h2>
Suppose genetic sequences are arrays of symbols. Only five symbols are possible (<tt>A</tt>, <tt>C</tt>, <tt>G</tt>, <tt>T</tt> and <tt>N</tt>). When comparing two sequences, we account for only four symbols (<tt>A</tt>, <tt>C</tt>, <tt>G</tt> and <tt>T</tt>) and ignore <tt>N</tt>-symbols. The performance of an algorithm is the priority compared to the storage size. Therefore, for each symbol, we may allocate four bits, so <tt>A = 1000</tt>, <tt>C = 0100</tt>, <tt>G = 0010</tt>, <tt>T = 0001</tt> and <tt>N = 0000</tt>. As SIMD instructions often deal with 128-bit chunks of data, we may split genetic sequences into 32-symbol chunks and store the data corresponding to each 32-symbol chunk as 128-bit data. We store each of the four bits for symbols in a separate 32-bit data block. Let us have the following sequence of 32 symbols: <tt>CATAGNCACGTGATCCTAGNCATGTTACCTGT</tt>. We may store this array as

<table>
  <tr>
    <th>m</th>
    <th><tt>CATAGNCACGTGATCCTAGNCATGTTACCTGT</tt></th>
    <th></th>
  </tr>
  <tr>
    <th><i>A</i></th>
    <th><tt>01010001000010000100010000100000</tt></th>
    <th><tt>0x0422108a</tt></th>
  </tr>
  <tr>
    <th><i>C</i></th>
    <th><tt>10000010100000110000100000011000</tt></th>
    <th><tt>0x1810c141</tt></th>
  </tr>
  <tr>
    <th><i>G</i></th>
    <th><tt>00001000010100000010000100000010</tt></th>
    <th><tt>0x40840a10</tt></th>
  </tr>
  <tr>
    <th><i>T</i></th>
    <th><tt>00100000001001001000001011000101</tt></th>
    <th><tt>0xa3412404</tt></th>
    </tr>
  <tr>
    <th><i>A|C|G|T</i></th>
    <th><tt>11111011111111111110111111111111</tt></th>
    <th><tt>0xfff7ffdf</tt></th>
  </tr>
  </table>

We may set the above letter using <a href="https://software.intel.com/sites/landingpage/IntrinsicsGuide/">Intel Intrinsics</a>

<p>
  <tt>__m128i m1 = _mm_set_epi32(0xa3412404, 0x40840a10, 0x1810c141, 0x0422108a);</tt>
</p>

<h2 id="link_seed">Spaced seeds</h2>

Usually spaced seeds are <b>binary</b> ones. For example, we have a seed <tt>1101011101</tt>, <tt>1</tt>-elements mean that we should account for the corresponding elements of a genetic sequence; otherwise, we ignore it. So, for a sequence <tt>ACAGTCCATG</tt> of the same length, we get <tt>AC_G_CCA_G</tt> or after removing spaces, we obtain <tt>ACGCCAG</tt>. Therefore, the seed <tt>1101011101</tt> forms a signature <tt>ACGCCAG</tt>, which can be written as a number after we replace symbols with bits. 

We may shuffle elements of the signature and form other signatures, i.e. <tt>CGACCGA</tt> or <tt>GCCCGAA</tt>. The order of symbols is not essential when we form a signature, and we need to keep the shuffling procedure the same for all genetic sequences we use. The main goal is to make the shuffling procedure as fast as possible so there are as few SIMD operations as possible. 


<h2 id="link_ternary">Ternary spaced seeds</h2>

Binary seeds are good for general sequences. However, in genetics, the chance of pointwise mutations is different for pairs of symbols. Therefore there are <b>transition</b> (<tt>A &#x27F7; G</tt>, <tt>C &#x27F7; T</tt>) and <b>transversion</b> mutations (<tt>A &#x27F7; C</tt>, <tt>A &#x27F7; T</tt>, <tt>G &#x27F7; C</tt>, <tt>G &#x27F7; T</tt>). So, instead of binary seeds, we may use <b>ternary</b> ones. We keep the same notations as in <a href="https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-5-149">this paper</a>: <tt>_</tt> (do not care symbol), <tt>#</tt> (match), <tt>@</tt> (transition mismatch, i.e. when symbols <tt>A</tt> and <tt>G</tt>, <tt>C</tt> and <tt>T</tt> considered as the same symbols).


<h2 id="link_form">Forming signatures</h2>

Suppose there is a binary seed of length <i>L</i> and weight <i>w</i> (number of 1-elements), and we apply it to a given genetic sequence of the same length. Using four-bits-per-symbol representation, we get <i>4*w</i> bits. We write this data as several 128-bit blocks, padded by <tt>N</tt>-elements to a multiple of 32 symbols. As <tt>N</tt>-elements will be ignored when a signature is calculated, then instead of four bits per symbol, we may use only two (after we have checked that there are no <tt>N</tt>-symbols in the shuffled subsequence). Therefore, we may form two arrays of bits (<tt>A|C</tt> and <tt>A|G</tt>). These calculations are easy to perform with SIMD logical and bit shift operations.

In the case of ternary seeds, we need to create one array of bits (<tt>A|G</tt>) for <tt>@</tt>-elements. So, for a given ternary seed we form two binary seeds: 1) replace <tt>&#x5f;</tt> and <tt>@</tt> by <tt>0</tt>, <tt>#</tt> by <tt>1</tt>, 2) replace <tt>&#x5f;</tt> and <tt>#</tt> by <tt>0</tt>, <tt>@</tt> by <tt>1</tt>. Let their weights be <i>w<sub>#</sub></i> and <i>w<sub>@</sub></i>. By definition, the total weight is <i>w = w<sub>#</sub> + w<sub>@</sub>/2</i> and the number of bits required to store the signature is the double weight, i.e. <i>2w<sub>#</sub> + w<sub>@</sub></i>. So, #-seed provides us with two arrays (<tt>A|C</tt> and <tt>A|G</tt>) each of length <i>w<sub>#</sub></i>, and @-seed with one array (<tt>A|G</tt>) of length <i>w<sub>@</sub></i>.

If there are more than 32 symbols, we must concatenate corresponding bits into one <i>2w</i>-bit number.

<h2 id="link_tools">Tools</h2>

To compile the codes one can use oneAPI <a href="https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html">compiler</a> 

OpenMP is used. Possible command line for compilation

<div>icpc combitesBinary.cpp -std=c++17 -qopenmp -o combitesB.exe</div>

Please uncomment WIN32 definition if you compile for Windows OS (different functions to allocate aligned memeory are used).

Example output files can be found in the Examples folder.

