%%
% Algoritmo Genético aplicado à função Rastrigin
% Nome: Thiago Silva <tsprates@hotmail.com>
% Data: 05/10/2017
%
% Uso:
%	[x, f, P] = AG(ncal, nvar)
%
% Sendo:
%	x: vetor das variáveis de decisão do melhor indivíduo;
%	f: valor da função objetivo avaliada em x*;
%	P: valor da função de penalidade avaliada em x*;
%	ncal: número total de chamadas da função de cálculo da fitness do problema;
%	nvar: número de variáveis de decisão.
%
function [ x, f, P ] = AG(ncal, nvar)
% Número de indivíduos
Npop = 10 * nvar;

% Taxa de cruzamento
Pc = 0.9;

% Número de avaliações da função-objetivo
Fo = 1;

% GERAÇÃO DA POPULAÇÃO INICIAL
Pop = GeracaoInicial(Npop, nvar + 4);

% População-seleção
SelPop = zeros(Npop, nvar + 4);

% AVALIAÇÃO
for K = 1:Npop
    X = Pop(K,:);
    Fit = Fitness(X, nvar, 1, ncal);
    Pop(K, end-3:end) = Fit;
    Fo = Fo + 1;
end

% CRITÉRIO DE PARADA
while Fo <= ncal
    
    % ELITISMO
    [~, Pos] = min(Pop(:, end));
    Melhor = Pop(Pos, :);
    
    % SELEÇÃO
    for I=1:Npop
        if rand() < 0.5
            Sel = Roleta(Pop);
        else
            Sel = Torneio(Pop, Npop);
        end
        SelPop(I, :) = Pop(Sel, :);
    end
    
    % Embaralha soluções
    Rands = randperm(Npop);
    
    I = 1;
    while I <= Npop
        if rand() < Pc
            % CRUZAMENTO
            Pai1 = SelPop(Rands(I), :);
            Pai2 = SelPop(Rands(mod(I, Npop) + 1), :);
            
            OpCr = randi(3);
            if OpCr == 1
                [Filho1, Filho2] = CruzamentoPolarizado(Pai1, Pai2, nvar);
            elseif OpCr == 2
                [Filho1, Filho2] = SBX(Pai1, Pai2, nvar);
            else
                % Cruzamento Proposto
                [Filho1, Filho2] = CruzamentoUniforme(Pai1, Pai2, nvar);
            end
            
            % Adiciona filhos à nova população
            if (Npop-I) > 1
                Pop(I, :) = Filho1;
                Pop(I + 1, :) = Filho2;
                I = I + 2;
            else
                if rand() < 0.5
                    Pop(I, :) = Filho1;
                else
                    Pop(I, :) = Filho2;
                end
                I = I + 1;
            end
        else
            % MUTAÇÃO
            OpMut = randi(3);
            if OpMut == 1
                Mut = MutacaoReal(SelPop(Rands(I), :), nvar);
            elseif OpMut == 2
                Mut = MutacaoPolinomial(SelPop(Rands(I), :), nvar);
            else
                % Mutação Proposta
                Mut = MutacaoGaussiana(SelPop(Rands(I), :));
            end
            
            % Adiciona mutação à nova população
            Pop(I, :) = Mut;
            I = I + 1;
        end
    end
    
    % ELITISMO
    [~, Pos] = max(Pop(:, end));
    Pop(Pos, :) = Melhor;
    
    % AVALIAÇÃO
    for K = 1:Npop
        X = Pop(K, :);
        Fit = Fitness(X, nvar, Fo, ncal);
        Pop(K, end-3:end) = Fit;
        Fo = Fo + 1;
    end
    
    % Controle de diversidade
    Fmin = min(Pop(:, end));
    Fmedia = mean(Pop(:, end));
    Mdg = (Fmedia - Fmin) / Fmedia;
    if Mdg < 0.4
        Pc = Pc * 1.2;
    end
    if Mdg > 0.7
        Pc = Pc / 1.2;
    end
end

% Vetor das variáveis de decisão do melhor indivíduo x*
x = Melhor(1:end-4);

% Valor da função objetivo avaliada em x*
f = Melhor(end-3);

% Valor da função de penalidade avaliada em x*
P = Melhor(end-1) + Melhor(end-2);
end


%%
% Cálculo de fitness
function [ F ] = Fitness( X, Nvar , Cal, Ncal )
g = 0;
h = 0;

NC = Cal / Ncal;
if NC < 0.6
    R = 0;
    S = 0;
elseif NC < 0.9
    R = NC * 10;
    S = NC * 10;
else
    R = NC * 100;
    S = NC * 100;
end

f = 10 * Nvar;
for i = 1:Nvar
    f = f + (X(i)^2 - 10*cos(2*pi*X(i)));
    
    gi = (sin(2*pi*X(i)) + 0.5);
    if gi > 0
        g = g + gi^2;
    end
    
    hi = (cos(2*pi*X(i)) + 0.5);
    if hi ~= 0
        h = h + hi^2;
    end
end

F(1) = f;
F(2) = g;
F(3) = h;
F(4) = f + R*F(2) + S*F(3);
end


%%
% Geração da população inicial
function [ Pop ] = GeracaoInicial(Npop, Nvar)
Pop = zeros(Npop, Nvar);
Rands = rand(Npop, Nvar);
for i = 1:Npop
    for n = 1:Nvar
        Pop(i, n) = (5.12 - (-5.12)) * Rands(i, n) - 5.12;
    end
end
end


%%
% Seleção por Roleta
function [ IdxSel ] = Roleta( Pop )
TotalFit = sum(Pop(:, end));
Roleta = rand();
IdxSel = 1;
Soma = Pop(IdxSel, end) / TotalFit;
while Soma < Roleta
    IdxSel = IdxSel + 1;
    Soma = Soma + (Pop(IdxSel, end) / TotalFit);
end
end


%%
% Seleção por Torneio
function [ IdxSel ] = Torneio( Pop, Npop )
Sel = randperm(Npop, 2);
K = 0.75;
if rand() < K
    % Escolhe o melhor
    if Pop(Sel(1), end) < Pop(Sel(2), end)
        IdxSel = Sel(1);
    else
        IdxSel = Sel(2);
    end
else
    % Escolhe o pior
    if Pop(Sel(1), end) > Pop(Sel(2), end)
        IdxSel = Sel(1);
    else
        IdxSel = Sel(2);
    end
end
end


%%
% Cruzamento Binário Simulado (SBX) (Deb & Agrawal, 1994)*
function [ F1, F2 ] = SBX(Xi, Xj, Nvar)
F1 = zeros(1, length(Xi));
F2 = zeros(1, length(Xj));
Eta = 2;

for I = 1:Nvar
    u = rand();
    if u <= 0.5
        beta = (2*u) ^ (1/(Eta+1));
    else
        beta = (2*(1-u)) ^ -(1/(Eta+1));
    end
    
    F1(I) = 0.5*((1 + beta)*Xi(I) + (1 - beta)*Xj(I));
    F2(I) = 0.5*((1 - beta)*Xi(I) + (1 + beta)*Xj(I));
end
end


%%
% Cruzamento Polarizado e Linear
function [ F1, F2 ] = CruzamentoPolarizado(Pai1, Pai2, Nvar)
Alfapol = 0.9;
Alfa = 0.5;
Kcross = randi(Nvar);

F1 = Pai1;
F2 = Pai2;

if rand() <= 0.5
    F1(Kcross:end) = F1(Kcross:end) * Alfapol + F2(Kcross:end) * Alfa;
    F2(Kcross:end) = F2(Kcross:end) * Alfa + F2(Kcross:end) * Alfapol;
else
    F1(1:Kcross) = F1(1:Kcross) * Alfapol + F2(1:Kcross) * Alfa;
    F2(1:Kcross) = F2(1:Kcross) * Alfa + F2(1:Kcross) * Alfapol;
end
end


%%
% Cruzamento Uniforme
function [ F1, F2 ] = CruzamentoUniforme(Pai1, Pai2, Nvar)
F1 = Pai1;
F2 = Pai2;

Mascara = round(rand(Nvar, 1));
for K = 1:Nvar
    if Mascara(K) == 1
        F1(K) = Pai2(K);
        F2(K) = Pai1(K);
    end
end
end


%%
% Mutação (Codificação Real) {Qing et al, Meneguim, Sérgio}
function [ Gama ] = MutacaoReal(X, Nvar)
Gama = X;
Kmut = randi(Nvar);
Beta = 2*rand() - 1;
Range = (5.12 - (-5.12));
Dir = rand();

if Dir < 0.5
    Gama(1:Kmut) = 0.05 * Beta * Range;
else
    Gama(Kmut:end) = 0.05 * Beta * Range;
end
end


%%
% Mutação Polinomial (Deb & Goyal, 1996)*
function [ M ] = MutacaoPolinomial(X, Nvar)
Eta = 5;
Sigma = 0.4;
V = zeros(1, length(X));

for K = 1:Nvar
    u = rand();
    if u <= 0.5
        Delta = (2*u)^(1/(Eta+1)) - 1;
    else
        Delta = 1 - (2*(1-u))^(1/(Eta+1));
    end
    
    V(K) = Sigma * (5.12 - (-5.12)) * Delta;
end
M = X + V;
end


%%
% Mutação Gaussiana
function [ M ] = MutacaoGaussiana( X )
M = X;
Val = normrnd(0, (5.12-(-5.12)) * 0.1);
M = M + Val;
end