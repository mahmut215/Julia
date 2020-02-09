using Pkg; for p in ("Knet","ArgParse"); haskey(Pkg.installed(),p) || Pkg.add(p); end

module MLP
using Knet,ArgParse
include(Knet.dir("data","mnist.jl"))

function predict(w,x)
    atype=Array{Float32}
    forward = Any[]
    # println("size(w) :",size(w))
    # println("size(x) :",size(x))
    for i=1:2:length(w)
        x =mat(x)*w[i] .+ w[i+1]
        x = sigm.(x) # max(0,x)
        push!(forward, convert(atype,x))
    end
    # println("forward : ",forward)
    # println("size(forward) :",size(forward))
    return forward
end

function decoder(w,latent)
    atype=Array{Float32}
    # println("latent :",latent)
    # println("w length(w) :",length(w))
    data = Any[]
    for i=length(w)-1:-2:1
        # println("size(w) :",i,size(w[i]))
        latent =mat(latent)*transpose(w[i]) .+ w[i+1]

        if i<length(w)-1
        # x = sigm.(x) # max(0,x)
            push!(data, convert(atype,latent))
        # println("xx : ",size(forward))
        end
    end
    # println("data : ",data[length(data)])
    # println("size(forward) :",size(forward))
    return data
end

function accuracy(forward,ytr)
    ind=length(forward)
    # println("ind : ",ind)
    cost =(ytr-forward[ind]).^2

    cost=round.(cost, digits=2)
    # println("cost : ",cost)
    cost=findall(x->x==0.00, cost)
    numOfdata=length(ytr)
    r=length(cost)
    # println("num of true : ",r)
    # println("numOfdata : ",numOfdata)
    yuzde =100/numOfdata
    # println("cost : ",cost)
    # println("----accuracy : ",yuzde*r)
    return yuzde*r
    # println("forward : ",forward[ind])
end

function dataAnd()
    xtrn=cat(dims=4, [0.1,0.3], [1.0,0.0], [0.0,1.0], [1.0,1.0])
    xtrn=Float32.(xtrn)
    ytrn=Array{UInt8}([0;1;1;1])
    xtst=cat(dims=4, [0.9 0.8],[1.2 0.1],[0.1 0.8],[0.2 0.1])
    xtst=Float32.(xtst)
    ytst=Array{UInt8}([1;1;1;0])

    return xtrn,ytrn,xtst,ytst
end

function weights(h...; atype=Array{Float32}, winit=0.1)
    w = Any[]
    # w1=convert(Array{Float32},[0.5 0.62; 0.1 0.2]');
    # println("w1 : ",size(w1))
    # w2=convert(Array{Float32},[0.4 -0.1]')
    # # println(size(w2))
    # w2=transpose(w2)
    # println("w2 : ",size(w2))
    # w3=convert(Array{Float32},[-0.2; 0.3]')
    # w3=transpose(w3)
    # println("w3 : ",size(w3))
    # # println(size(w3))
    # w4=convert(Array{Float32},[1.83]')
    # # println(size(w4))
    # println("w4 : ",size(w4))
    # push!(w, convert(atype, w1))
    # push!(w, convert(atype, w2))
    # push!(w, convert(atype, w3))
    # push!(w, convert(atype, w4))
    x = 2
    # println("x : h ",x)
    for y in [h..., 1]
        # println("y : ",y)
        push!(w, convert(atype, winit*randn(x,y)))
        push!(w, convert(atype, zeros(1, y)))
        x = y
        # println("b start : ",w[2])
    end
    # for i=1:length(w)
    #     println(i," : ",w[i])
    # end
    return w
end


function reshapeData(x,y,batchsize; shuffle=false,partial=false,xtype=typeof(x),ytype=typeof(y),xsize=size(x), ysize=size(y))
    nx = size(x)[end]
    if nx != size(y)[end]; throw(DimensionMismatch()); end
    x2 = reshape(x, :, nx)
    y2 = reshape(y, :, nx)
    x2=transpose(x2)
    y2=transpose(y2)
    return x2,y2
end
function derivative(x)
    return x .*(1.0 .- x)
end


function backword(w,forward,ytr)
    derivat = Any[]
    atype=Array{Float32}
    ind=length(forward)
    # println("ind : ",ind)
    error =ytr-forward[ind]
    # println("error : ",error)
    x=0
    "kısmi türevi. Bahsedilen zincir kuralı "
    for i=1:1:length(forward)
        push!(derivat, convert(atype, derivative(forward[i])))
        # derivat=derivative(forward[i])
    end
    d_back = Any[]
    derivat_len = length(derivat)
    for i=length(w)+1:-2:1
        if i>length(w)
            x= error .* derivat[derivat_len]
            derivat_len=derivat_len-1
            push!(d_back, convert(atype,x))
            # println("d_back : ",d_back)
        elseif i>1 && i<length(w)
            x =mat(x)*transpose(w[i])

            push!(d_back, convert(atype,x.* derivat[derivat_len]))
            derivat_len=derivat_len-1
            # println("d_back " ,d_back)
        end
    end
    # println("d_back " ,d_back)
    return d_back
end

function updateW(w, xtrn,lr,forward,backwod)
    # println("backwod : ",backwod)
    # println("----------------------")
    ind=1
    ind2=length(backwod)
    ind3=length(backwod)-1
    for i=1:1:length(w)
        "birinci agirlik guncellemek icin w[1] "
        if i==1
            w[i]+=(transpose(xtrn)*backwod[length(backwod)]).*lr
        "bais agirliklari guncellemek icin "
        elseif i%2==0
            w[i]+=sum(backwod[ind2],dims=1).*lr
            ind2=ind2-1
        "hidden agirliklari guncellemek icin "
        elseif i%2!=0 && i>1
            w[i]+=(transpose(forward[ind])*backwod[ind3]).*lr
            ind=ind+1
            ind3=ind3-1
        end
    end
    # println("size(w): ",size(w))
    return w
end

function main(args="")
    s = ArgParseSettings()
    s.description="mlp.jl (c) Deniz Yuret, 2016. Multi-layer perceptron model on the MNIST handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=1; help="minibatch size")
        ("--epochs"; arg_type=Int; default=10; help="number of epochs for training")
        ("--rapor"; arg_type=Int; default=10; help="How after num of epochs you want rapor")
        ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--lr"; arg_type=Float64; default=0.1; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients per parameter")
        # These are to experiment with sparse arrays
        # ("--xtype"; help="input array type: defaults to atype")
        # ("--ytype"; help="output array type: defaults to atype")
    end
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    o = parse_args(args, s; as_symbols=true)
    if !o[:fast]
        println(s.description)
        println("opts=",[(k,v) for (k,v) in o]...)
    end
    o[:seed] > 0 && Knet.seed!(o[:seed])
    atype = eval(Meta.parse(o[:atype]))
    w = weights(o[:hidden]...; atype=atype, winit=o[:winit])
    # xtrn,ytrn,xtst,ytst = mnist()
    xtrn,ytrn,xtst,ytst = dataAnd()
    # global dtrn = minibatch(xtrn, ytrn, o[:batchsize]; xtype=atype)
    # global dtst = minibatch(xtst, ytst, o[:batchsize]; xtype=atype)
    xtrn,ytrn=reshapeData(xtrn, ytrn, o[:batchsize]; xtype=atype)
    xtst,ytst=reshapeData(xtst, ytst, o[:batchsize]; xtype=atype)
    # println("xxx: ",size(ytrn))
    # report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn,predict),:tst,accuracy(w,dtst,predict)))
    # if o[:fast]
    #     (train(w, dtrn; lr=o[:lr], epochs=o[:epochs]); gpu()>=0 && Knet.cudaDeviceSynchronize())
    # else
    #     report(0)

        @time for epoch=1:o[:epochs]
            # println("epoch : ",epoch)
            "predict function ( ileri yayılım algoritması )"
            forward =predict(w,xtrn)
            # println("forward " ,forward[])

            " Geri Yayılım Algoritması "
            backwod=backword(w,forward,ytrn)

            lr=o[:lr]
            # println("lr :",lr)
            "Agirlikleri guncelleme "
            w=updateW(w, xtrn,lr,forward,backwod)

            "xtst data testi"
            forwardtest =predict(w,xtst)
            # # println((:tst,accuracy(forward,ytst)))
            "rapor yazdirma "
            if epoch%o[:rapor]==0
                println((:epoch,epoch,:trn,accuracy(forward,ytrn),:tst,accuracy(forwardtest,ytst)))
            end
        end
        forward =predict(w,xtst)
        println("test data output neron: ", forward[length(forward)])
        println("-----------------------------------------")
        forward =predict(w,xtrn)
        println("xtrn data output neron: ", forward[length(forward)])
        println("-----------------------------------------")
    "agirliklari son hali yazdirmak icin alttaki  yorumlari kaldirin"
    # for i=1:length(w)
    #     println(i,"w End : ",w[i])
    # end
    println("-----------------------------------------")
    " Yeni data olusturma (decoder) "
    newData=decoder(w,forward[length(forward)])
    println("new data after train (Yeni data olusturma (decoder)): ")
    println(newData[length(newData)])
println("-----------------------------------------")
    "test yeni data dogru mu degil mi"
    forward =predict(w,newData[length(newData)])
    println("new data testing: ", forward[length(forward)])
    println("-----------------------------------------")
end
PROGRAM_FILE == "mlp.jl" && main(ARGS)

end # module
