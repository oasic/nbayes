require 'yaml'

unless Hash.new.respond_to?(:default_proc=)
  class Hash
    def default_proc=(proc)
      initialize(&proc)
    end
  end
end

# == NBayes::Base
#
# Robust implementation of NaiveBayes:
# - using log probabilities to avoid floating point issues
# - Laplacian smoothing for unseen tokens
# - allows binarized or standard NB
# - allows Prior distribution on category to be assumed uniform (optional)
# - generic to work with all types of tokens, not just text
#

module NBayes

  class Base

    attr_accessor :assume_uniform, :debug, :k, :vocab, :data, :log_vocab
    attr_reader :binarized

    def initialize(options={})
      @debug = false
      @k = 1
      @binarized = options[:binarized] || false
      @log_vocab = false            # for smoothing, use log of vocab size, rather than vocab size
      @assume_uniform = false
      @vocab = Hash.new             # used to calculate vocab size (@vocab.keys.length)
      @data = Hash.new
      @data.default_proc = get_default_proc()
      #@data = {
      #  "category1": {
      #    "tokens": Hash.new(0),
      #    "total_tokens": 0,
      #    "examples": 0
      #  },
      # ...
      #}
    end

    # Allows removal of low frequency words that increase processing time and may overfit
    # - tokens with a count less than x (measured by summing across all classes) are removed
    # Ex: nb.purge_less_than(2)
    #
    # NOTE: this does not decrement the "examples" count, so purging is not *always* the same
    # as if the item was never added in the first place, but usually so
    def purge_less_than(x)
      remove_list = {}
      @vocab.keys.each do |token|
        count = @data.keys.inject(0){|sum, cat| sum + @data[cat][:tokens][token] }
        next if count >= x
        @data.each do |cat, cat_data|
          count = cat_data[:tokens][token]
          cat_data[:tokens].delete(token)     # delete and retrieve count
          cat_data[:total_tokens] -= count    # subtract that count from cat counts
        end  # each category hash
        # print "removing #{token}\n"
        remove_list[token]=1
      end  # each vocab word
      remove_list.keys.each {|token| @vocab.delete(token) }
      # print "total vocab size is now #{vocab_size}\n"
    end

    # Returns the default proc used by the data hash
    # Separate method so that it can be used after data import
    def get_default_proc
      return lambda do |hash, category|
        hash[category] = {
          :tokens => Hash.new(0),             # holds freq counts
          :total_tokens => 0,
          :examples => 0
        }
      end
    end

    # called internally after yaml import to reset Hash defaults
    def reset_after_import
      @data.default_proc = get_default_proc()
      @data.each {|cat, cat_hash| cat_hash[:tokens].default = 0 }
    end

    def ham(tokens)
      train(tokens, 'ham')
    end

    def spam(tokens)
      train(tokens, 'spam')
    end

    def train(tokens, category)
      cat_data = @data[category]
      cat_data[:examples] += 1
      tokens = tokens.uniq if binarized
      tokens.each do |w|
        @vocab[w] = 1
        cat_data[:tokens][w] += 1
        cat_data[:total_tokens] += 1
      end
    end

    def classify(tokens)
      print "classify: #{tokens.join(', ')}\n" if @debug
      probs = {}
      tokens = tokens.uniq  if binarized
      probs = calculate_probabilities(tokens)
      print "results: #{probs.to_yaml}\n" if @debug
      probs.extend(NBayes::Result)
      probs
    end

    # Total number of training instances
    def total_examples
      sum = 0
      @data.each {|cat, cat_data| sum += cat_data[:examples] }
      sum
    end

    # Returns the size of the "vocab" - the number of unique tokens found in the text
    # This is used in the Laplacian smoothing.
    def vocab_size
      return Math.log(@vocab.keys.length) if @log_vocab
      @vocab.keys.length
    end

    def category_stats
      tmp = []
      data.each do |cat, data|
        e = data[:examples]
        t = data[:total_tokens]
        tmp << "For category #{cat}, %d examples (%.02f%% of the total) and %d total_tokens" % [e, 100.0 * e / total_examples, t]
      end
      tmp.join("\n")
    end

    # Calculates the actual probability of a class given the tokens
    # (this is the work horse of the code)
    def calculate_probabilities(tokens)
      # P(class|words) = P(w1,...,wn|class) * P(class) / P(w1,...,wn)
      #                = argmax P(w1,...,wn|class) * P(class)
      #
      # P(wi|class) = (count(wi, class) + k)/(count(w,class) + kV)
      prob_numerator = {}
      v_size = vocab_size

      cat_prob = Math.log(1 / @data.count.to_f)
      example_count = total_examples.to_f

      @data.keys.each do |category|
        cat_data = @data[category]

        unless assume_uniform
          cat_prob = Math.log(cat_data[:examples] / example_count)
        end

        log_probs = 0
        cat_denominator = (cat_data[:total_tokens]+ @k * v_size).to_f
        tokens.each do |token|
          log_probs += Math.log( (cat_data[:tokens][token] + @k) / cat_denominator )
        end
        prob_numerator[category] = log_probs + cat_prob
      end
      # calculate the denominator, which normalizes this into a probability; it's just the sum of all numerators from above
      normalizer = 0
      prob_numerator.each {|cat, numerator| normalizer += numerator }
      # One more caveat:
      # We're using log probabilities, so the numbers are negative and the smallest negative number is actually the largest prob.
      # To convert, we need to maintain the relative distance between all of the probabilities:
      # - divide log prob by normalizer: this keeps ratios the same, but reverses the ordering
      # - re-normalize based off new counts
      # - final calculation
      # Ex: -1,-1,-2  =>  -4/-1, -4/-1, -4/-2
      #   - renormalize and calculate => 4/10, 4/10, 2/10
      intermed = {}
      renormalizer = 0
      prob_numerator.each do |cat, numerator|
        intermed[cat] = normalizer / numerator.to_f
        renormalizer += intermed[cat]
      end
      # calculate final probs
      final_probs = {}
      intermed.each do |cat, value|
        final_probs[cat] = value / renormalizer.to_f
      end
      final_probs
    end

    def self.from_yml(yml_data)
      nbayes = YAML.load(yml_data)
      nbayes.reset_after_import()  # yaml does not properly set the defaults on the Hashes
      nbayes
    end

    # Loads class instance from a data file (e.g., yaml)
    def self.from(yml_file)
      File.open(yml_file, "rb") do |file|
        self.from_yml(file.read)
      end
    end

    # Load class instance
    def load(yml)
      if yml.nil?
        nbayes = NBayes::Base.new
      elsif yml[0..2] == "---"
        nbayes = self.class.from_yml(yml)
      else
        nbayes = self.class.from(yml)
      end
      nbayes
    end

    # Dumps class instance to a data file (e.g., yaml) or a string
    def dump(arg)
      if arg.instance_of? String
        File.open(arg, "w") {|f| YAML.dump(self, f) }
      else
        YAML.dump(arg)
      end
    end

  end


  module Result
    def max_class
      keys.max{ |a,b| self[a] <=> self[b] }
    end
  end

end
