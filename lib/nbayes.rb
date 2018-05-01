require 'yaml'
require_relative 'dbconnection'

# == NBayes::Base
#
# Robust implementation of NaiveBayes:
# - using log probabilities to avoid floating point issues
# - Laplacian smoothing for unseen tokens
# - allows binarized or standard NB
# - allows Prior distribution on category to be assumed uniform (optional)
# - generic to work with all types of tokens, not just text


module NBayes

  class Vocab
    attr_accessor :log_size, :tokens

    def initialize(options = {})
      @tokens = DBConnection.new
      # for smoothing, use log of vocab size, rather than vocab size
      @log_size = options[:log_size]
    end

    def delete(token)
      tokens.delete(token)
    end

    def each(&block)
      tokens.keys.each(&block)
    end

    def size
      if log_size
        Math.log(tokens.count)
      else
        tokens.count
      end
    end

    def seen_token(token)
      tokens[token] = 1
    end
  end

  class Data
    attr_accessor :data
    def initialize(options = {})
      @data = Hash.new
      #@data = {
      #  "category1": {
      #    "tokens": Hash.new(0),
      #    "total_tokens": 0,
      #    "examples": 0
      #  },
      # ...
      #}
    end

    def categories
      data.keys
    end

    def token_trained?(token, category)
      data[category] ? data[category][:tokens].has_key?(token) : false
    end

    def cat_data(category)
      unless data[category].is_a? Hash # change to value of DBConnection object
        data[category] = new_category
      end
      data[category]
    end

    def category_stats
      tmp = []
      total_example_count = total_examples
      self.each do |category|
        e = example_count(category)
        t = token_count(category)
        tmp << "For category #{category}, %d examples (%.02f%% of the total) and %d total_tokens" % [e, 100.0 * e / total_example_count, t]
      end
      tmp.join("\n")
    end

    def each(&block)
      data.keys.each(&block)
    end

    # Increment the number of training examples for this category
    def increment_examples(category)
      cat_data(category)[:examples] += 1
    end

    # Decrement the number of training examples for this category.
    # Delete the category if the examples counter is 0.
    def decrement_examples(category)
      cat_data(category)[:examples] -= 1
      delete_category(category) if cat_data(category)[:examples] < 1
    end

    def example_count(category)
      cat_data(category)[:examples]
    end

    def token_count(category)
      cat_data(category)[:total_tokens]
    end

    # XXX - Add Enumerable and see if I get inject?
    # Total number of training instances
    def total_examples
      sum = 0
      self.each {|category| sum += example_count(category) }
      sum
    end

    # Add this token to this category
    def add_token_to_category(category, token)
      cat_data(category)[:tokens][token] += 1
      cat_data(category)[:total_tokens] += 1
    end

    # Decrement the token counter in a category
    # If the counter is 0, delete the token.
    # If the total number of tokens is 0, delete the category.
    def remove_token_from_category(category, token)
      cat_data(category)[:tokens][token] -= 1
      delete_token_from_category(category, token) if cat_data(category)[:tokens][token] < 1
      cat_data(category)[:total_tokens] -= 1
      delete_category(category) if cat_data(category)[:total_tokens] < 1
    end

    # How many times does this token appear in this category?
    def count_of_token_in_category(category, token)
      cat_data(category)[:tokens][token]
    end

    def delete_token_from_category(category, token)
      count = count_of_token_in_category(category, token)
      cat_data(category)[:tokens].delete(token)
      # Update this category's total token count
      cat_data(category)[:total_tokens] -= count
    end

    def purge_less_than(token, x)
      return if token_count_across_categories(token) >= x
      self.each do |category|
        delete_token_from_category(category, token)
      end
      true  # Let caller know we removed this token
    end

    # XXX - TODO - use count_of_token_in_category
    # Return the total number of tokens we've seen across all categories
    def token_count_across_categories(token)
      data.keys.inject(0){|sum, cat| sum + @data[cat][:tokens][token] }
    end

    def reset_after_import
      self.each {|category| cat_data(category)[:tokens].default = 0 }
    end

    def new_category
      {
        :tokens => Hash.new(0),             # holds freq counts
        :total_tokens => 0,
        :examples => 0
      }
    end

    def delete_category(category)
      data.delete(category) if data.has_key?(category)
      categories
    end

  end

  class Base

    attr_accessor :assume_uniform, :debug, :k, :vocab, :data
    attr_reader :binarized

    def initialize(options={})
      @debug = false
      @k = 1
      @binarized = options[:binarized] || false
      @assume_uniform = false
      @vocab = Vocab.new(:log_size => options[:log_vocab])
      @data = Data.new
    end

    # Allows removal of low frequency words that increase processing time and may overfit
    # - tokens with a count less than x (measured by summing across all classes) are removed
    # Ex: nb.purge_less_than(2)
    #
    # NOTE: this does not decrement the "examples" count, so purging is not *always* the same
    # as if the item was never added in the first place, but usually so
    def purge_less_than(x)
      remove_list = {}
      @vocab.each do |token|
        if data.purge_less_than(token, x)
          # print "removing #{token}\n"
          remove_list[token] = 1
        end
      end  # each vocab word
      remove_list.keys.each {|token| @vocab.delete(token) }
      # print "total vocab size is now #{vocab.size}\n"
    end

    # Delete an entire category from the classification data
    def delete_category(category)
      data.delete_category(category)
    end

    def train(tokens, category)
      tokens = tokens.uniq if binarized
      data.increment_examples(category)
      tokens.each do |token|
        vocab.seen_token(token)
        data.add_token_to_category(category, token)
      end
    end

    # Be carefull with this function:
    # * It decrement the number of examples for the category.
    #   If the being-untrained category has no more examples, it is removed from the category list.
    # * It untrain already trained tokens, non existing tokens are not considered.
    def untrain(tokens, category)
      tokens = tokens.uniq if binarized
      data.decrement_examples(category)

      tokens.each do |token|
        if data.token_trained?(token, category)
          vocab.delete(token)
          data.remove_token_from_category(category, token)
        end
      end
    end

    def classify(tokens)
      print "classify: #{tokens.join(', ')}\n" if @debug
      probs = {}
      tokens = tokens.uniq if binarized
      probs = calculate_probabilities(tokens)
      print "results: #{probs.to_yaml}\n" if @debug
      probs.extend(NBayes::Result)
      probs
    end

    def category_stats
      data.category_stats
    end

    # Calculates the actual probability of a class given the tokens
    # (this is the work horse of the code)
    def calculate_probabilities(tokens)
      # P(class|words) = P(w1,...,wn|class) * P(class) / P(w1,...,wn)
      #                = argmax P(w1,...,wn|class) * P(class)
      #
      # P(wi|class) = (count(wi, class) + k)/(count(w,class) + kV)
      prob_numerator = {}
      v_size = vocab.size

      cat_prob = Math.log(1 / data.categories.count.to_f)
      total_example_count = data.total_examples.to_f

      data.each do |category|
        unless assume_uniform
          cat_prob = Math.log(data.example_count(category) / total_example_count)
        end

        log_probs = 0
        denominator = (data.token_count(category) + @k * v_size).to_f
        tokens.each do |token|
          numerator = data.count_of_token_in_category(category, token) + @k
          log_probs += Math.log( numerator / denominator )
        end
        prob_numerator[category] = log_probs + cat_prob
      end
      normalize(prob_numerator)
    end

    def normalize(prob_numerator)
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

    # called internally after yaml import to reset Hash defaults
    def reset_after_import
      data.reset_after_import
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
    # Return the key having the largest value
    def max_class
      keys.max{ |a,b| self[a] <=> self[b] }
    end
  end

end
