require File.expand_path(File.dirname(__FILE__) + '/spec_helper')

describe "RobustNBayes" do
  before do
    @nbayes = RobustNBayes.new
  end

  it "should assign equal probability to each class" do
    @nbayes.train( %w[a b c d e f g], 'classA' ) 
    @nbayes.train( %w[a b c d e f g], 'classB' ) 
    results = @nbayes.classify( %w[a b c] )
    results['classA'].should == 0.5
    results['classB'].should == 0.5
  end

  it "should handle more than 2 classes" do
    @nbayes.train( %w[a a a a], 'classA' ) 
    @nbayes.train( %w[b b b b], 'classB' ) 
    @nbayes.train( %w[c c], 'classC' ) 
    results = @nbayes.classify( %w[a a a a b c] )
    results.max_class.should == 'classA'
    results['classA'].should >= 0.4
    results['classB'].should <= 0.3
    results['classC'].should <= 0.3
  end

  it "should use smoothing by default to eliminate errors w/division by zero" do
    @nbayes.train( %w[a a a a], 'classA' ) 
    @nbayes.train( %w[b b b b], 'classB' ) 
    results = @nbayes.classify( %w[x y z] )
    results['classA'].should >= 0.0
    results['classB'].should >= 0.0
  end

  it "should optionally purge low frequency data" do
    100.times do
      @nbayes.train( %w[a a a a], 'classA' ) 
      @nbayes.train( %w[b b b b], 'classB' ) 
    end
    @nbayes.train( %w[a], 'classA' ) 
    @nbayes.train( %w[c b], 'classB' ) 
    results = @nbayes.classify( %w[c] )
    results.max_class.should == 'classB' 
    results['classB'].should > 0.5 
    @nbayes.data['classB'][:tokens]['c'].should == 1

    @nbayes.purge_less_than(2)			# this removes the entry for 'c' in 'classB' because it has freq of 1

    results = @nbayes.classify( %w[c] )
    @nbayes.data['classB'][:tokens]['c'].should == 0
    results['classA'].should == 0.5 
    results['classB'].should == 0.5 
  end

  it "should allow uniform class distribution" do
  end

  it "should allow log of vocab size in smoothing" do
  end

  it "should allow binarized mode" do
  end

  it "should save to yaml" do
  end

  it "should load from yaml" do
  end

  #it "fails" do
  #  fail "hey buddy, you should probably rename this file and start specing for real"
  #end
end
