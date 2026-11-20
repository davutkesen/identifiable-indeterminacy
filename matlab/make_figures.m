function make_figures(outdir)
%MAKE_FIGURES  The four figures of the article, written as Fig1.eps ... Fig4.eps.
%
%   make_figures            writes the files into the folder figures/
%   make_figures(outdir)    writes them into outdir
%
%   The function writes the PostScript of each figure itself and does not use
%   the plotting functions of MATLAB.  figprolog.ps, which is copied into
%   every file, holds the two fonts and the drawing procedures; the function
%   adds the coordinates.  Every coordinate is a whole number of thousandths
%   of a point and is computed in integer arithmetic, so the files come out
%   the same, byte for byte, on every machine and in every release.  After
%   writing, each file is compared with the figure of the article through the
%   sizes and checksums listed in reference.txt.
%
%   Every figure is 337 pt wide, the 119 mm that the journal asks for to the
%   nearest point, and is meant to be printed at that size: all lettering is
%   8 pt in a Helvetica-compatible sans serif that is embedded in the file,
%   no line is thinner than 0.5 pt, and the bounding box is the figure size.
%
%     1  direct truth and proposed support over a two-feature grid
%     2  the components of both constructions along one feature axis
%     3  cross-validated macro F1 against the indeterminacy weight
%     4  the regime relation: 133 tasks of sources A to C and 53 PMLB tasks
%
%   Figures 1 and 2 need no data.  Figures 3 and 4 read data/fig3_scores.csv
%   and data/fig4_tasks.csv, whose columns are described in README.txt.

    here = fileparts(mfilename('fullpath'));
    if nargin < 1
        outdir = fullfile(here, 'figures');
    end
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end
    prolog = fileread(fullfile(here, 'figprolog.ps'));
    prolog(prolog == 13) = [];
    while prolog(end) == 10
        prolog(end) = [];
    end
    fig1(prolog, outdir);
    fig2(prolog, outdir);
    fig3(prolog, outdir, readrows(fullfile(here, 'data', 'fig3_scores.csv'), 6));
    fig4(prolog, outdir, readrows(fullfile(here, 'data', 'fig4_tasks.csv'), 4));
    compare_with_reference(outdir, fullfile(here, 'reference.txt'));
end


% ------------------------------------------------------------------ numbers
function mu = centroids()
% The two class centroids on every feature, in tenths.
    mu = [3, 7];
end


function q = rdiv(a, b)
% The quotient a / b of two whole numbers, rounded to the nearest one.
    if b < 0
        a = -a;
        b = -b;
    end
    q = floor((2 * a + b) / (2 * b));
end


function [d0, d1, m] = distances(u, g)
% The point u / g of one feature: its distances d0 and d1 to the two class
% centroids and the smaller of the two, m, in units of 1 / g.  The direct
% truth there is (g - m) / g.  In the proposed construction the
% indeterminacy is m / g, and the supports of the two classes are
% (g - m) / g times d1 / (d0 + d1) and d0 / (d0 + d1).
    MU = centroids();
    d0 = abs(u - MU(1) * g / 10);
    d1 = abs(u - MU(2) * g / 10);
    m = min(d0, d1);
end


function rows = readrows(fn, ncol)
% The whole numbers of a comma-separated file with one header line.
    s = fileread(fn);
    nl = find(s == 10, 1);
    v = sscanf(strrep(s(nl + 1:end), ',', ' '), '%f');
    rows = reshape(v, ncol, []).';
end


% --------------------------------------------------------------------- text
function s = up(t)
    s = sprintf('[0 8000 0 (%s)]', t);          % upright, 8 pt
end


function s = it(t)
    s = sprintf('[1 8000 0 (%s)]', t);          % italic
end


function s = sb(t)
    s = sprintf('[0 5600 -1300 (%s)]', t);      % upright subscript
end


function s = isb(t)
    s = sprintf('[1 5600 -1300 (%s)]', t);      % italic subscript
end


function s = th()
    s = '[0 4800 0 ( )]';                       % thin space
end


function s = txt(varargin)
    s = sprintf('%s ', varargin{:});
    s = ['[' s(1:end - 1) ']'];
end


% ------------------------------------------------------------ building blocks
function L = header(name, wd, ht, prolog)
    L = {'%!PS-Adobe-3.0 EPSF-3.0', ...
         sprintf('%%%%BoundingBox: 0 0 %d %d', wd, ht), ...
         sprintf('%%%%Title: %s', name), ...
         '%%Creator: make_figures', ...
         '%%LanguageLevel: 2', ...
         '%%DocumentData: Clean7Bit', ...
         '%%EndComments', ...
         '%%BeginProlog', ...
         prolog, ...
         '%%EndProlog', ...
         '0.001 0.001 scale'};
end


function save_eps(L, out, name)
    L = [L, {'showpage', '%%EOF'}];
    fid = fopen(fullfile(out, [name '.eps']), 'w');
    fwrite(fid, uint8(sprintf('%s\n', L{:})));
    fclose(fid);
    fprintf('wrote %s.eps\n', name);
end


function out = wrap(v)
% Whole numbers, twelve to a line.
    out = {};
    for i = 1:12:numel(v)
        out{end + 1} = strtrim(sprintf('%d ', v(i:min(i + 11, numel(v)))));
    end
end


function out = polyline(x0, xs, y0, ys)
    out = {sprintf('N %d %d M', x0 + xs(1), y0 + ys(1))};
    for i = 2:numel(xs)
        out{end + 1} = sprintf('%d %d L', x0 + xs(i), y0 + ys(i));
    end
    out{end + 1} = 'S';
end


function out = xaxis(x, y, wd, num, den, labels)
% Ticks below an axis at the fractions num / den of its width.
    out = {};
    for i = 1:numel(num)
        X = x + rdiv(num(i) * wd, den);
        out{end + 1} = sprintf('%d %d %d %d SG', X, y, X, y - 3000);
        if ~isempty(labels)
            out{end + 1} = sprintf('%d %d 1 %s T', X, y - 12300, txt(up(labels{i})));
        end
    end
end


function out = yaxis(x, y, ht, num, den, labels)
% Ticks left of an axis at the fractions num / den of its height.
    out = {};
    for i = 1:numel(num)
        Y = y + rdiv(num(i) * ht, den);
        out{end + 1} = sprintf('%d %d %d %d SG', x, Y, x - 3000, Y);
        if ~isempty(labels)
            out{end + 1} = sprintf('%d %d 2 %s T', x - 6500, Y - 2900, txt(up(labels{i})));
        end
    end
end


function [a, b] = feature_value(kind, u, g)
% The value of one feature at the point u / g, as the fraction a / (g b):
% the direct truth for kind 1, the support of class 1 in the proposed
% construction for kind 2.
    [d0, d1, m] = distances(u, g);
    if kind == 1
        a = g - m;
        b = 1;
    else
        a = (g - m) * d0;
        b = d0 + d1;
    end
end


function r = crossing(kind, i, a, b, c, k, S, first)
% Where a contour cuts the edge between the nodes i and i + 1 of one axis,
% the other coordinate being a node with the value a / (c b): the distance
% from node i in units of 1 / S, found by bisection.  first tells whether
% the mean of the two values is at least k / 10 at node i.  Nodes are
% counted from 0.
    h = S / c;
    lo = 0;
    hi = h;
    while hi - lo > 1
        mid = floor((lo + hi) / 2);
        [au, bu] = feature_value(kind, i * h + mid, S);
        if (10 * c * au * b + 10 * S * a * bu - 2 * k * c * S * bu * b >= 0) == first
            lo = mid;
        else
            hi = mid;
        end
    end
    if first
        r = lo;
    else
        r = hi;
    end
end


function out = contour_lines(kind, c, k, x0, y0, S)
% The line on which a panel of Figure 1 takes the value k / 10, as line
% segments in a square of side S with the lower left corner (x0, y0).
%
% The value at (x1, x2) is the mean of the values of the two features.  It
% is compared with the level at the (c + 1) x (c + 1) nodes of a grid, and
% each cell is cut where the line crosses its edges (marching squares);
% G tells where the value is at least the level.
    h = S / c;
    A = zeros(1, c + 1);
    B = zeros(1, c + 1);
    for u = 0:c
        [A(u + 1), B(u + 1)] = feature_value(kind, u, c);
    end
    G = false(c + 1, c + 1);
    for j = 1:c + 1
        for i = 1:c + 1
            G(j, i) = 10 * (A(i) * B(j) + A(j) * B(i)) - 2 * k * c * B(i) * B(j) >= 0;
        end
    end
    out = {};
    for j = 1:c
        for i = 1:c
            g00 = G(j, i);
            g10 = G(j, i + 1);
            g01 = G(j + 1, i);
            g11 = G(j + 1, i + 1);
            px = [];
            py = [];
            if g00 ~= g10                                % lower edge
                px(end + 1) = x0 + (i - 1) * h + crossing(kind, i - 1, A(j), B(j), c, k, S, g00);
                py(end + 1) = y0 + (j - 1) * h;
            end
            if g10 ~= g11                                % right edge
                px(end + 1) = x0 + i * h;
                py(end + 1) = y0 + (j - 1) * h + crossing(kind, j - 1, A(i + 1), B(i + 1), c, k, S, g10);
            end
            if g01 ~= g11                                % upper edge
                px(end + 1) = x0 + (i - 1) * h + crossing(kind, i - 1, A(j + 1), B(j + 1), c, k, S, g01);
                py(end + 1) = y0 + j * h;
            end
            if g00 ~= g01                                % left edge
                px(end + 1) = x0 + (i - 1) * h;
                py(end + 1) = y0 + (j - 1) * h + crossing(kind, j - 1, A(i), B(i), c, k, S, g00);
            end
            if numel(px) == 2
                pairs = [1, 2];
            elseif numel(px) == 4
                % all four edges are cut: the value at the centre of the
                % cell decides which corners belong together
                ctr = 10 * (A(i) * B(i + 1) * B(j) * B(j + 1) + A(i + 1) * B(i) * B(j) * B(j + 1) ...
                            + A(j) * B(i) * B(i + 1) * B(j + 1) + A(j + 1) * B(i) * B(i + 1) * B(j)) ...
                    - 4 * k * c * B(i) * B(i + 1) * B(j) * B(j + 1);
                if g00 == (ctr >= 0)
                    pairs = [1, 2, 3, 4];
                else
                    pairs = [1, 4, 2, 3];
                end
            else
                pairs = [];
            end
            for t = 1:2:numel(pairs)
                a = pairs(t);
                b = pairs(t + 1);
                if px(a) ~= px(b) || py(a) ~= py(b)
                    out{end + 1} = sprintf('%d %d %d %d SG', px(a), py(a), px(b), py(b));
                end
            end
        end
    end
end


% ------------------------------------------------------------------ figure 1
function fig1(prolog, out)
    L = header('Fig1', 337, 171, prolog);
    MU = centroids();
    K = 1000000;                     % values are carried as whole numbers of 1 / K
    S = 120960;                      % side of a panel
    X0 = [36000, 169200];            % left edges of the two panels
    Y0 = 29520;                      % their lower edge
    n = 500;                         % the image of a panel has n x n pixels
    c = 120;                         % its contours are traced on c x c cells
    ttl = {txt(up('\(a\) direct truth '), it('T')), ...
           txt(up('\(b\) proposed support '), it('M'), sb('1'))};
    lab = {'0', '0.25', '0.5', '0.75', '1'};
    for p = 1:2
        % The value at (x1, x2) is the mean over the two features of a
        % per-feature value: the direct truth in (a), the support of class 1
        % in (b).  The image takes it at the centres of the pixels.
        prof = zeros(1, n);
        for i = 1:n
            [a, b] = feature_value(p, 2 * i - 1, 2 * n);
            prof(i) = rdiv(a * K, 2 * n * b);
        end
        L{end + 1} = sprintf('%d %d %d %d %d {', X0(p), Y0, S, S, n);
        L = [L, wrap(prof)];
        L{end + 1} = sprintf('} %d HM', K);
        % contours at 0.1, 0.2, ..., 0.9
        L{end + 1} = '255 255 255 rgb 500 setlinewidth [] 0 setdash 1 setlinecap 1 setlinejoin';
        for k = 1:9
            L = [L, contour_lines(p, c, k, X0(p), Y0, S)];
        end
        % the centroids, and in (a) the two mixed points
        a0 = rdiv(MU(1) * S, 10);
        a1 = rdiv(MU(2) * S, 10);
        L{end + 1} = '1300 setlinewidth 0 setlinecap 0 setlinejoin';
        L{end + 1} = sprintf('%d %d 3000 MOo', X0(p) + a0, Y0 + a0);
        L{end + 1} = sprintf('%d %d 3000 MSo', X0(p) + a1, Y0 + a1);
        if p == 1
            L{end + 1} = '232 38 42 rgb 1600 setlinewidth';
            L{end + 1} = sprintf('%d %d 3500 MX', X0(p) + a0, Y0 + a1);
            L{end + 1} = sprintf('%d %d 3500 MX', X0(p) + a1, Y0 + a0);
        end
        L{end + 1} = '0 0 0 rgb 600 setlinewidth';
        L{end + 1} = sprintf('%d %d %d %d RS', X0(p), Y0, S, S);
        L = [L, xaxis(X0(p), Y0, S, [0, 1, 2, 3, 4], 4, lab)];
        if p == 1
            L = [L, yaxis(X0(p), Y0, S, [0, 1, 2, 3, 4], 4, lab)];
        else
            L = [L, yaxis(X0(p), Y0, S, [0, 1, 2, 3, 4], 4, {})];
        end
        L{end + 1} = sprintf('%d %d 1 %s T', X0(p) + S / 2, Y0 + S + 4200, ttl{p});
        L{end + 1} = sprintf('%d %d 1 %s T', X0(p) + S / 2, Y0 - 22000, txt(it('x'), sb('1')));
    end
    L{end + 1} = sprintf('%d %d 90 1 %s TR', 8700, Y0 + S / 2, txt(it('x'), sb('2')));
    % the colour scale
    XB = 296640;
    WB = 6120;
    lab = {'0', '0.2', '0.4', '0.6', '0.8', '1'};
    L{end + 1} = sprintf('%d %d %d %d CB', XB, Y0, WB, S);
    L{end + 1} = sprintf('%d %d %d %d RS', XB, Y0, WB, S);
    for i = 1:6
        Y = Y0 + rdiv((i - 1) * S, 5);
        L{end + 1} = sprintf('%d %d %d %d SG', XB + WB, Y, XB + WB + 3000, Y);
        L{end + 1} = sprintf('%d %d 0 %s T', XB + WB + 6500, Y - 2900, txt(up(lab{i})));
    end
    L{end + 1} = sprintf('%d %d 90 1 %s TR', 328700, Y0 + S / 2, txt(up('membership value')));
    save_eps(L, out, 'Fig1');
end


% ------------------------------------------------------------------ figure 2
function fig2(prolog, out)
    L = header('Fig2', 337, 166, prolog);
    MU = centroids();
    PW = 142560;                     % width and height of a panel
    PH = 117360;
    X0 = [36000, 190080];
    Y0 = 28800;
    g = 500;                         % the curves are drawn at x = u / g
    % A value v sits at the fraction (v + 0.03) / 1.53 of the height.
    xs = zeros(1, g + 1);
    yT = zeros(1, g + 1);
    yF = zeros(1, g + 1);
    yI = zeros(1, g + 1);
    yM0 = zeros(1, g + 1);
    yM1 = zeros(1, g + 1);
    for u = 0:g
        [d0, d1, m] = distances(u, g);
        xs(u + 1) = rdiv(u * PW, g);
        yT(u + 1) = rdiv((100 * (g - m) + 3 * g) * PH, 153 * g);
        yF(u + 1) = rdiv((max(0, 100 * m - 22 * g) + 3 * g) * PH, 153 * g);
        yI(u + 1) = rdiv((100 * m + 3 * g) * PH, 153 * g);
        yM0(u + 1) = rdiv((100 * (g - m) * d1 + 3 * g * (d0 + d1)) * PH, 153 * g * (d0 + d1));
        yM1(u + 1) = rdiv((100 * (g - m) * d0 + 3 * g * (d0 + d1)) * PH, 153 * g * (d0 + d1));
    end
    yc = rdiv((22 + 3) * PH, 153);           % the constant I = 0.22 of the direct construction
    y1 = rdiv((100 + 3) * PH, 153);          % the sum of the proposed components, 1 everywhere
    sT = '196 78 82 rgb 1500 setlinewidth [] 0 setdash';
    sI = '129 114 179 rgb 1500 setlinewidth [5550 2400] 0 setdash';
    sF = '147 120 96 rgb 1500 setlinewidth [1500 2475] 0 setdash';
    sM0 = '221 132 82 rgb 1500 setlinewidth [] 0 setdash';
    sM1 = '76 114 176 rgb 1500 setlinewidth [9600 2400 1500 2400] 0 setdash';
    s1 = '0 0 0 rgb 900 setlinewidth [900 1485] 0 setdash';
    ttl = {txt(up('\(a\) direct')), txt(up('\(b\) proposed'))};
    lab = {'0', '0.2', '0.4', '0.6', '0.8', '1'};
    for p = 1:2
        x = X0(p);
        top = Y0 + PH;
        L{end + 1} = sprintf('gsave %d %d %d %d rectclip', x, Y0, PW, PH);
        L{end + 1} = '179 179 179 rgb 600 setlinewidth [] 0 setdash 0 setlinecap 1 setlinejoin';
        for t = 1:2
            X = x + rdiv(MU(t) * PW, 10);
            L{end + 1} = sprintf('%d %d %d %d SG', X, Y0, X, top);
        end
        if p == 1
            L{end + 1} = sT;
            L = [L, polyline(x, xs, Y0, yT)];
            L{end + 1} = sI;
            L{end + 1} = sprintf('%d %d %d %d SG', x, Y0 + yc, x + PW, Y0 + yc);
            L{end + 1} = sF;
            L = [L, polyline(x, xs, Y0, yF)];
        else
            L{end + 1} = sI;
            L = [L, polyline(x, xs, Y0, yI)];
            L{end + 1} = sM0;
            L = [L, polyline(x, xs, Y0, yM0)];
            L{end + 1} = sM1;
            L = [L, polyline(x, xs, Y0, yM1)];
            L{end + 1} = s1;
            L{end + 1} = sprintf('%d %d %d %d SG', x, Y0 + y1, x + PW, Y0 + y1);
        end
        L{end + 1} = 'grestore';
        % legend: two columns of line samples
        r1 = top - 11100;
        r2 = top - 21500;
        if p == 1
            c2 = x + 93280;
            L{end + 1} = sT;
            L{end + 1} = sprintf('%d %d %d %d SG', x + 22680, r1 + 2800, x + 37880, r1 + 2800);
            L{end + 1} = sI;
            L{end + 1} = sprintf('%d %d %d %d SG', x + 22680, r2 + 2800, x + 37880, r2 + 2800);
            L{end + 1} = sF;
            L{end + 1} = sprintf('%d %d %d %d SG', c2, r1 + 2800, c2 + 15200, r1 + 2800);
            L{end + 1} = '0 0 0 rgb';
            L{end + 1} = sprintf('%d %d 0 %s T', x + 44280, r1, txt(it('T')));
            L{end + 1} = sprintf('%d %d 0 %s T', x + 44280, r2, txt(it('I'), up(' \(constant\)')));
            L{end + 1} = sprintf('%d %d 0 %s T', c2 + 21600, r1, txt(it('F')));
        else
            c2 = x + 62280;
            L{end + 1} = sI;
            L{end + 1} = sprintf('%d %d %d %d SG', x + 22680, r1 + 2800, x + 37880, r1 + 2800);
            L{end + 1} = sM0;
            L{end + 1} = sprintf('%d %d %d %d SG', x + 22680, r2 + 2800, x + 37880, r2 + 2800);
            L{end + 1} = sM1;
            L{end + 1} = sprintf('%d %d %d %d SG', c2, r1 + 2800, c2 + 15200, r1 + 2800);
            L{end + 1} = s1;
            L{end + 1} = sprintf('%d %d %d %d SG', c2, r2 + 2800, c2 + 15200, r2 + 2800);
            L{end + 1} = '0 0 0 rgb';
            L{end + 1} = sprintf('%d %d 0 %s T', x + 44280, r1, txt(it('I')));
            L{end + 1} = sprintf('%d %d 0 %s T', x + 44280, r2, txt(it('M'), sb('0')));
            L{end + 1} = sprintf('%d %d 0 %s T', c2 + 21600, r1, txt(it('M'), sb('1')));
            L{end + 1} = sprintf('%d %d 0 %s T', c2 + 21600, r2, ...
                                 txt(it('I'), th(), up('+'), th(), up('\202'), isb('c'), th(), ...
                                     it('M'), isb('c'), th(), up('\201'), th(), up('1')));
        end
        L{end + 1} = '0 0 0 rgb 600 setlinewidth [] 0 setdash 0 setlinejoin';
        L{end + 1} = sprintf('%d %d %d %d RS', x, Y0, PW, PH);
        L = [L, xaxis(x, Y0, PW, [0, 1, 2, 3, 4, 5], 5, lab)];
        if p == 1
            L = [L, yaxis(x, Y0, PH, [3, 23, 43, 63, 83, 103], 153, lab)];
        else
            L = [L, yaxis(x, Y0, PH, [3, 23, 43, 63, 83, 103], 153, {})];
        end
        L{end + 1} = sprintf('%d %d 1 %s T', x + PW / 2, top + 4200, ttl{p});
        L{end + 1} = sprintf('%d %d 1 %s T', x + PW / 2, Y0 - 22000, txt(it('x'), isb('j')));
    end
    L{end + 1} = sprintf('%d %d 90 1 %s TR', 14200, Y0 + PH / 2, txt(up('component value')));
    save_eps(L, out, 'Fig2');
end


% ------------------------------------------------------------------ figure 3
function fig3(prolog, out, rows)
% rows: panel (1 direct, 2 proposed), the selected k, the indeterminacy
% weight and the two other weights in tenths (T and F in the direct
% construction, M0 and M1 in the proposed one), and the cross-validated
% macro F1 on dipeptide composition in units of 1 / 1000000.
    L = header('Fig3', 337, 187, prolog);
    PW = 142560;
    PH = 138960;
    X0 = [36000, 190080];
    Y0 = 28800;
    MK = {'MO', 'MS', 'MT', 'MD', 'MV'};
    boxc = {'139 47 50 rgb', '47 79 143 rgb'};
    boxx = [37, 35];                 % centres of the two notes, at
    boxy = [41, 52];                 % (0.33, 0.56) and (0.31, 0.67)
    name = {txt(up('share on '), it('T')), txt(up('share on '), it('M'), sb('0'))};
    lab = {'0', '0.2', '0.4', '0.6', '0.8', '1'};
    labv = {'0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'};
    for p = 1:2
        x = X0(p);
        top = Y0 + PH;
        sel = rows(rows(:, 1) == p, :);
        % A weight w sits at the fraction (w + 0.04) / 1.04 of the width, a
        % score s at the fraction (s - 0.15) / 0.65 of the height.
        L{end + 1} = '220 220 220 rgb 500 setlinewidth [] 0 setdash 0 setlinecap';
        for t = 0:5
            X = x + rdiv((4 + 20 * t) * PW, 104);
            L{end + 1} = sprintf('%d %d %d %d SG', X, Y0, X, top);
        end
        for t = 0:6
            Y = Y0 + rdiv((5 + 10 * t) * PH, 65);
            L{end + 1} = sprintf('%d %d %d %d SG', x, Y, x + PW, Y);
        end
        % One curve for every share of the remaining weight that occurs at
        % three or more grid points; the share is kept in units of 1 / 10000.
        key = zeros(size(sel, 1), 1);
        for r = 1:size(sel, 1)
            if sel(r, 4) + sel(r, 5) > 0
                key(r) = rdiv(10000 * sel(r, 4), sel(r, 4) + sel(r, 5));
            else
                key(r) = -1;
            end
        end
        shares = [];
        u = unique(key).';
        for t = u
            if t >= 0 && sum(key == t) >= 3
                shares(end + 1) = t;
            end
        end
        nq = numel(shares);
        flat = 0;
        L{end + 1} = sprintf('gsave %d %d %d %d rectclip', x, Y0, PW, PH);
        for q = 1:nq
            pts = sortrows(sel(key == shares(q), [3, 6]));
            xs = zeros(1, size(pts, 1));
            ys = zeros(1, size(pts, 1));
            for i = 1:size(pts, 1)
                xs(i) = rdiv((10 * pts(i, 1) + 4) * PW, 104);
                ys(i) = rdiv((pts(i, 2) - 150000) * PH, 650000);
            end
            flat = max(flat, max(pts(:, 2)) - min(pts(:, 2)));
            L{end + 1} = sprintf('%d vir 1100 setlinewidth [] 0 setdash 0 setlinecap 1 setlinejoin', ...
                                 floor(256 * (8 * (nq - 1) + 84 * (q - 1)) / (100 * (nq - 1))));
            L = [L, polyline(x, xs, Y0, ys)];
            for i = 1:numel(xs)
                L{end + 1} = sprintf('%d %d 1600 %s', x + xs(i), Y0 + ys(i), MK{mod(q - 1, 5) + 1});
            end
        end
        if p == 2
            % the selected weights: the highest score of the whole grid
            best = sel(1, :);
            for r = 1:size(sel, 1)
                if sel(r, 6) > best(6)
                    best = sel(r, :);
                end
            end
            L{end + 1} = '232 38 42 rgb 1300 setlinewidth 0 setlinejoin';
            L{end + 1} = sprintf('%d %d 5500 MSTAR', x + rdiv((10 * best(3) + 4) * PW, 104), ...
                                 Y0 + rdiv((best(6) - 150000) * PH, 650000));
        end
        L{end + 1} = 'grestore';
        t = rdiv(flat, 1000);
        L{end + 1} = sprintf('%s 600 setlinewidth [] 0 setdash 0 setlinejoin', boxc{p});
        L{end + 1} = sprintf('%d %d 9000 2500 [%s %s] BT', ...
                             x + rdiv(boxx(p) * PW, 104), Y0 + rdiv(boxy(p) * PH, 65), ...
                             txt(up('largest variation')), ...
                             txt(up(sprintf('along any curve: %d.%03d', floor(t / 1000), mod(t, 1000)))));
        % legend
        for q = 1:nq
            yb = Y0 + 7300 + (nq - q) * 9110;
            h = rdiv(shares(q), 100);
            L{end + 1} = sprintf('%d vir 1100 setlinewidth', ...
                                 floor(256 * (8 * (nq - 1) + 84 * (q - 1)) / (100 * (nq - 1))));
            L{end + 1} = sprintf('%d %d %d %d SG', x + PW - 44000, yb + 2800, x + PW - 31200, yb + 2800);
            L{end + 1} = sprintf('%d %d 1600 %s', x + PW - 37600, yb + 2800, MK{mod(q - 1, 5) + 1});
            L{end + 1} = '0 0 0 rgb';
            L{end + 1} = sprintf('%d %d 0 %s T', x + PW - 24800, yb, ...
                                 txt(up(sprintf('%d.%02d', floor(h / 100), mod(h, 100)))));
        end
        L{end + 1} = sprintf('%d %d 1 %s T', x + PW - 26600, Y0 + 7300 + nq * 9110 + 300, name{p});
        L{end + 1} = '0 0 0 rgb 600 setlinewidth';
        L{end + 1} = sprintf('%d %d %d %d RS', x, Y0, PW, PH);
        L = [L, xaxis(x, Y0, PW, [4, 24, 44, 64, 84, 104], 104, lab)];
        if p == 1
            L = [L, yaxis(x, Y0, PH, [5, 15, 25, 35, 45, 55, 65], 65, labv)];
            L{end + 1} = sprintf('%d %d 1 %s T', x + PW / 2, top + 4200, ...
                                 txt(up('\(a\) direct, '), it('k'), th(), up('='), th(), ...
                                     up(sprintf('%d', sel(1, 2)))));
        else
            L = [L, yaxis(x, Y0, PH, [5, 15, 25, 35, 45, 55, 65], 65, {})];
            L{end + 1} = sprintf('%d %d 1 %s T', x + PW / 2, top + 4200, ...
                                 txt(up('\(b\) proposed, '), it('k'), th(), up('='), th(), ...
                                     up(sprintf('%d', sel(1, 2)))));
        end
        L{end + 1} = sprintf('%d %d 1 %s T', x + PW / 2, Y0 - 22000, ...
                             txt(up('indeterminacy weight '), it('w'), isb('I')));
    end
    L{end + 1} = sprintf('%d %d 90 1 %s TR', 14200, Y0 + PH / 2, txt(up('cross-validated macro F1')));
    save_eps(L, out, 'Fig3');
end


% ------------------------------------------------------------------ figure 4
function fig4(prolog, out, rows)
% rows: panel, source (1 to 4 for A to D), and the two coordinates x and y of
% a task in units of 1 / 1000000.  Panel 1 is the fixed configuration, k = 5
% with uniform weights and coupled coordinates; panel 2 has both rules tuned
% and the coordinates cross-fitted, and adds the 53 tasks of source D, which
% played no part in finding the relation.
    L = header('Fig4', 337, 318, prolog);
    AW = 285360;                     % width and height of a panel
    PH = 112320;
    X0 = 44640;
    YB = [187200, 28800];            % lower edges of panels (a) and (b)
    w = 25;                          % tasks in the moving average
    MK = {'MO', 'MS', 'MT', 'MD'};
    sz = [1730, 1730, 1730, 1580];
    colour = {'76 114 176 rgb', '221 132 82 rgb', '85 168 104 rgb', '129 114 179 rgb'};
    lab = {'0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'};
    labv = {'\2000.4', '\2000.3', '\2000.2', '\2000.1', '0', '0.1', '0.2'};
    for p = 1:2
        Y0 = YB(p);
        top = Y0 + PH;
        sel = rows(rows(:, 1) == p, :);
        % A value x sits at the fraction (x + 0.03) / 0.73 of the width, a
        % value y at the fraction (y + 0.50) / 0.76 of the height.
        X1 = X0 + rdiv(13 * AW, 73);             % x = 0.10
        X3 = X0 + rdiv(33 * AW, 73);             % x = 0.30
        L{end + 1} = sprintf('237 242 249 rgb %d %d %d %d RF', X0, Y0, X1 - X0, PH);
        L{end + 1} = sprintf('251 238 238 rgb %d %d %d %d RF', X3, Y0, X0 + AW - X3, PH);
        L{end + 1} = '220 220 220 rgb 500 setlinewidth [] 0 setdash 0 setlinecap';
        for t = 0:7
            X = X0 + rdiv((3 + 10 * t) * AW, 73);
            L{end + 1} = sprintf('%d %d %d %d SG', X, Y0, X, top);
        end
        for t = 0:6
            Y = Y0 + rdiv((10 + 10 * t) * PH, 76);
            L{end + 1} = sprintf('%d %d %d %d SG', X0, Y, X0 + AW, Y);
        end
        L{end + 1} = '140 140 140 rgb 600 setlinewidth [600 1200] 0 setdash';
        L{end + 1} = sprintf('%d %d %d %d SG', X1, Y0, X1, top);
        L{end + 1} = sprintf('%d %d %d %d SG', X3, Y0, X3, top);
        Y = Y0 + rdiv(50 * PH, 76);
        L{end + 1} = '102 102 102 rgb 800 setlinewidth [] 0 setdash';
        L{end + 1} = sprintf('%d %d %d %d SG', X0, Y, X0 + AW, Y);
        L{end + 1} = sprintf('gsave %d %d %d %d rectclip', X0, Y0, AW, PH);
        for s = 1:4
            L{end + 1} = colour{s};
            for r = 1:size(sel, 1)
                if sel(r, 2) == s
                    L{end + 1} = sprintf('%d %d %d %s', X0 + rdiv((sel(r, 3) + 30000) * AW, 730000), ...
                                         Y0 + rdiv((sel(r, 4) + 500000) * PH, 760000), ...
                                         sz(s), MK{s});
                end
            end
        end
        % moving average of y over w tasks of sources A to C in the order of x
        xa = sel(sel(:, 2) <= 3, 3);
        ya = sel(sel(:, 2) <= 3, 4);
        [~, o] = sort(xa);
        mx = zeros(1, numel(xa) - w + 1);
        my = zeros(1, numel(xa) - w + 1);
        for i = 1:numel(xa) - w + 1
            tot = 0;
            for t = 0:w - 1
                tot = tot + ya(o(i + t));
            end
            mx(i) = rdiv((xa(o(i + floor(w / 2))) + 30000) * AW, 730000);
            my(i) = rdiv((tot + 500000 * w) * PH, 760000 * w);
        end
        L{end + 1} = '0 0 0 rgb 1400 setlinewidth [] 0 setdash 0 setlinecap 1 setlinejoin';
        L = [L, polyline(X0, mx, Y0, my)];
        L{end + 1} = 'grestore';
        L{end + 1} = '47 79 143 rgb';
        L{end + 1} = sprintf('%d %d 0 %s T', X0 + rdiv(8 * AW, 730), Y0 + rdiv(75 * PH, 76) - 5800, ...
                             txt(up('transformation helps')));
        L{end + 1} = '139 47 50 rgb';
        L{end + 1} = sprintf('%d %d 2 %s T', X0 + rdiv(722 * AW, 730), Y0 + rdiv(PH, 76) + 1700, ...
                             txt(up('transformation hurts')));
        % legend: (a) names the symbols of sources A to C, (b) adds source D
        yb = Y0 + 7700;
        if p == 1
            for s = 1:3
                L{end + 1} = colour{s};
                L{end + 1} = sprintf('%d %d %d %s', X0 + 12400 + 30950 * (s - 1), yb + 2800, sz(s), MK{s});
            end
            L{end + 1} = '0 0 0 rgb';
            L{end + 1} = sprintf('%d %d 0 %s T', X0 + 22000, yb, txt(up('A')));
            L{end + 1} = sprintf('%d %d 0 %s T', X0 + 52950, yb, txt(up('B')));
            L{end + 1} = sprintf('%d %d 0 %s T', X0 + 83900, yb, txt(up('C')));
            xl = X0 + 99300;
        else
            L{end + 1} = colour{4};
            L{end + 1} = sprintf('%d %d %d %s', X0 + 12400, yb + 2800, sz(4), MK{4});
            L{end + 1} = '0 0 0 rgb';
            L{end + 1} = sprintf('%d %d 0 %s T', X0 + 22000, yb, txt(up('D')));
            xl = X0 + 37400;
        end
        L{end + 1} = '1400 setlinewidth';
        L{end + 1} = sprintf('%d %d %d %d SG', xl, yb + 2800, xl + 12800, yb + 2800);
        L{end + 1} = sprintf('%d %d 0 %s T', xl + 16000, yb, txt(up('moving average, A to C')));
        L{end + 1} = '600 setlinewidth 0 setlinejoin';
        L{end + 1} = sprintf('%d %d %d %d RS', X0, Y0, AW, PH);
        L = [L, xaxis(X0, Y0, AW, [3, 13, 23, 33, 43, 53, 63, 73], 73, lab)];
        L = [L, yaxis(X0, Y0, PH, [10, 20, 30, 40, 50, 60, 70], 76, labv)];
        if p == 1
            L{end + 1} = sprintf('%d %d 1 %s T', X0 + AW / 2, top + 4200, ...
                                 txt(up('\(a\) '), it('k'), th(), up('='), th(), ...
                                     up('5, uniform weights')));
            L{end + 1} = sprintf('%d %d 1 %s T', X0 + AW / 2, Y0 - 22000, ...
                                 txt(it('x'), up(': macro F1 of plain '), it('k'), up('-NN \('), ...
                                     it('k'), th(), up('='), th(), ...
                                     up('5\) minus that of the majority classifier')));
        else
            L{end + 1} = sprintf('%d %d 1 %s T', X0 + AW / 2, top + 4200, ...
                                 txt(up('\(b\) both rules tuned, cross-fitted')));
            L{end + 1} = sprintf('%d %d 1 %s T', X0 + AW / 2, Y0 - 22000, ...
                                 txt(it('x'), up(': macro F1 of plain '), it('k'), up('-NN \('), ...
                                     it('k'), ...
                                     up(' tuned\) minus that of the majority classifier')));
        end
        L{end + 1} = sprintf('%d %d 90 1 %s TR', 8300, Y0 + PH / 2, txt(it('y'), up(': macro F1 gained')));
        L{end + 1} = sprintf('%d %d 90 1 %s TR', 17600, Y0 + PH / 2, txt(up('by the transformation')));
    end
    save_eps(L, out, 'Fig4');
end


% ------------------------------------------------------------------ checking
function compare_with_reference(outdir, reffile)
% Compares every file written with the figure of the article: reference.txt
% lists, for each figure, the size in bytes and the Adler-32 checksum of the
% file the article was typeset from.
    if ~exist(reffile, 'file')
        return
    end
    s = fileread(reffile);
    ref = reshape(sscanf(s(find(s == 10, 1) + 1:end), '%f'), 3, []).';
    for i = 1:size(ref, 1)
        fid = fopen(fullfile(outdir, sprintf('Fig%d.eps', ref(i, 1))), 'r');
        d = double(fread(fid, inf, 'uint8')).';
        fclose(fid);
        n = numel(d);
        a = mod(1 + sum(d), 65521);
        b = mod(n + sum((n:-1:1) .* d), 65521);
        if n == ref(i, 2) && b * 65536 + a == ref(i, 3)
            fprintf('Fig%d.eps is identical to the figure of the article (%d bytes)\n', ref(i, 1), n);
        else
            fprintf('Fig%d.eps DIFFERS from the figure of the article\n', ref(i, 1));
        end
    end
end
